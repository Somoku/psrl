import ray
import torch
from typing import Dict
from dataclasses import dataclass
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.utils.import_utils import load_extern_type
from verl.utils.dataset.rl_dataset import collate_fn, RLHFDataset


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 42))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


@dataclass
class DatasetType:
    """DataType for the dataset."""
    unknown: str = "unknown"
    train: str = "train"
    val: str = "val"
    test: str = "test"


@ray.remote
class DatasetHandle:
    def __init__(self, config, tokenizer, processor, rollout_instances_tp: Dict[int, int]) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.rollout_instances_tp = rollout_instances_tp # key: rollout instance id, value: tp size
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.train_dataloader_iter = None
        self.val_dataloader_iter = None
        
        # Initialize flags for each rollout instance
        # Write tag: 0 for train, 1 for val
        # Read tag: 0-tp indicates the number of times the rollout instance has been read for each tp rank
        self.rollout_instance_write_flag = {rollout_instance_id: DatasetType.unknown for rollout_instance_id in rollout_instances_tp.keys()}
        self.rollout_instance_read_flags = {rollout_instance_id: set() for rollout_instance_id in rollout_instances_tp.keys()}
        self.rollout_instance_cur_data = {rollout_instance_id: None for rollout_instance_id in rollout_instances_tp.keys()}
        
    def build_train_and_val_dataset(self) -> None:
        self.train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        self.val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        
    def build_train_sampler(self) -> None:
        assert self.train_dataset is not None, "Train dataset is not built yet. Call build_train_and_val_dataset() first."
        self.train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        
    def build_train_dataloader(self) -> None:
        assert self.train_dataset is not None, "Train dataset is not built yet. Call build_train_and_val_dataset() first."
        assert self.train_sampler is not None, "Train sampler is not built yet. Call build_train_sampler() first."
        batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        assert batch_size % self.config.psrl.deployment.n_rollout_instances == 0, f"Batch size {batch_size} is not divisible by the number of rollout instances {self.config.psrl.deployment.n_rollout_instances}"
        # TODO: use a smaller batch size, or different batch size for different rollout instance (with different parallel scheme)
        batch_size_per_instance = batch_size // self.config.psrl.deployment.n_rollout_instances
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size_per_instance,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=self.train_sampler, 
        )
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        
    def build_val_dataloader(self) -> None:
        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        print(f"Size of validation dataloader: {len(self.val_dataloader)}")
        
    def save_train_dataloader(self, dataloader_local_path: str) -> None:
        """Save the dataloader to a local path."""
        assert self.train_dataloader is not None, "Train dataloader is not built yet. Call build_train_dataloader() first."
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)
        print(f"Train dataloader saved to {dataloader_local_path}")
        
    def load_train_dataloader(self, dataloader_local_path: str) -> None:
        """Load the dataloader from a local path."""
        assert self.train_dataloader is not None, "Train dataloader is not built yet. Call build_train_dataloader() first."
        dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
        self.train_dataloader.load_state_dict(dataloader_state_dict)
        print(f"Train dataloader loaded from {dataloader_local_path}")
        
    def get_train_next(self):
        if self.train_dataloader_iter is None:
            self.train_dataloader_iter = iter(self.train_dataloader)
        epoch = 0
        try:
            data = next(self.train_dataloader_iter)
        except StopIteration:
            print("Train dataloader iterator exhausted.")
            epoch += 1
            if epoch == self.config.trainer.total_epochs:
                print("All epoches finished.")
                raise
            self.train_dataloader_iter = iter(self.train_dataloader)
            data = next(self.train_dataloader_iter)
        return data
    
    def get_val_next(self):
        if self.val_dataloader_iter is None:
            self.val_dataloader_iter = iter(self.val_dataloader)
        try:
            data = next(self.val_dataloader_iter)
        except StopIteration:
            print("Validation dataloader iterator exhausted.")
            self.val_dataloader_iter = iter(self.val_dataloader)
            raise
        return data
    
    def get_train_len(self):
        return len(self.train_dataloader)
    
    def get_val_len(self):
        return len(self.val_dataloader)
    
    # Get the current batch for a specific rollout instance
    # Only when all tp rank finish reading the data, the data will be updated (call get_next_data_func)
    def get_rollout_instance_batch_nowait(
        self,
        dataset_type: DatasetType,
        rollout_instance_id: int,
        rank: int
    ) :
        # Select the appropriate data-fetching function based on dataset type
        get_next_data_func = (
            self.get_train_next if dataset_type == DatasetType.train else self.get_val_next
        )

        # Write stage: fetch new data if the flag is unknown
        if self.rollout_instance_write_flag[rollout_instance_id] == DatasetType.unknown:
            assert not self.rollout_instance_read_flags[rollout_instance_id], (
                f"Rollout instance {rollout_instance_id}'s read_flags set is not empty: "
                f"{self.rollout_instance_read_flags[rollout_instance_id]}"
            )
            try:
                self.rollout_instance_cur_data[rollout_instance_id] = get_next_data_func()
            except StopIteration:
                print("get_rollout_instance_batch_nowait(): encountered StopIteration")
                raise
            # Mark the batch as ready for the current dataset type
            self.rollout_instance_write_flag[rollout_instance_id] = dataset_type

        # Read stage: if the batch is ready for this dataset type
        if self.rollout_instance_write_flag[rollout_instance_id] == dataset_type:
            ranks_set = self.rollout_instance_read_flags[rollout_instance_id]
            # If this rank already consumed the batch, treat as not-ready
            if rank in ranks_set:
                return None
            # Record this rank's consumption
            ranks_set.add(rank)
            # Once all TP ranks have read, reset flags for the next batch
            if len(ranks_set) == self.rollout_instances_tp[rollout_instance_id]:
                self.rollout_instance_write_flag[rollout_instance_id] = DatasetType.unknown
                ranks_set.clear()
            # Return the current batch data
            return self.rollout_instance_cur_data[rollout_instance_id]
        # If the batch is not ready, return None (waiting for another dataset type)
        return None
    
    # Get the current batch for the main controller
    # Just directly return the next batch   
    def get_single_controller_batch(self, dataset_type: DatasetType):
        get_next_data_func = self.get_train_next if dataset_type == DatasetType.train else self.get_val_next
        try:
            return get_next_data_func()
        except StopIteration:
            print("get_single_controller_batch() runs into StopIteration")
            raise