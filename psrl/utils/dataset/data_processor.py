import os
import numpy as np
import threading
import logging
from dataclasses import dataclass

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

import ray

from verl import DataProto

from psrl.utils.dataset.utils import create_rl_dataset, create_rl_sampler
from psrl.utils.logger import DualOutputHandler

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

@dataclass
class DatasetType:
    """DataType for the dataset."""
    unknown: str = "unknown"
    train: str = "train"
    val: str = "val"
    test: str = "test"


# NOTE(lhy): ray.remote must be declared here
# otherwise their will be weird bugs (NCCL broadcast/all-gather hangs, randomly crashed) during vllm generation
@ray.remote
class DataProcessor:
    def __init__(
        self,
        config,
        tokenizer,
        processor,
        ps_manager_handle,
        data_queue,
        collate_fn=None,
        process_mode="batch",
    ):
        """
        Initialize the DataProcessor, responsible for processing data batches.
        Note that this processor runs in a separate Ray worker on a single CPU.
        
        Args:
            config: Configuration object containing data processing parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            processor: Optional data processor, used for multimodal data
            ps_manager_handle: Parameter Server handle for communication with the PS worker.
            collate_fn: Function to collate data samples into batches.
            process_mode: Mode of processing data, either "stream" or "batch".
        """

        self.config = config
        
        # Dataset and dataloader attributes
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader_iter = None
        self.val_dataloader_iter = None
        self.global_steps = 0
        self._train_sample_idx = 0
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            self.collate_fn = default_collate_fn
        else:
            self.collate_fn = collate_fn
        
        # Communication handles
        self.ps_manager_handle = ps_manager_handle
        self.data_queue = data_queue
        
        self.process_mode = process_mode
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, \
            f"Rollout n {self.rollout_n} must be greater than or equal to alg_rollout_n {self.alg_rollout_n}."

        # Threads for data processing
        self.data_process_thread = None
        self.stop_data_process = False
        
        # Build logger
        self.log_prefix = "DataProcessor"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        
        # Create the initial datasets and dataloaders
        self.total_training_steps = None
        self._create_dataloader()

    # ------- Dataset and Dataloader Building Methods -------
    def build_train_and_val_dataset(self) -> None:
        """Build the training and validation datasets."""
        self.train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        self.val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        
    def build_train_sampler(self) -> None:
        """Build the training sampler."""
        assert self.train_dataset is not None, "Train dataset is not built yet. Call build_train_and_val_dataset() first."
        
        self.train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        
    def build_train_dataloader(self) -> None:
        """Build the training dataloader.

        This method creates a StatefulDataLoader for the training dataset.
        Note that the batch size is determined by `gen_batch_size` in the configuration, use `train_batch_size` as fallback.
        It also checks that the batch size is divisible by the number of rollout instances if process_mode is "batch".
        """
        assert self.train_dataset is not None, "Train dataset is not built yet. Call build_train_and_val_dataset() first."
        assert self.train_sampler is not None, "Train sampler is not built yet. Call build_train_sampler() first."

        if self.config.psrl.redundant_rollout.enable:
            batch_size = self.config.psrl.redundant_rollout.redundant_global_batch_size
        else:
            batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        
        assert self.config.psrl.gen_mode != "batch" or batch_size % self.config.psrl.deployment.n_rollout_instances == 0, \
            f"In batch mode, batch size {batch_size} is not divisible by" \
            f" the number of rollout instances {self.config.psrl.deployment.n_rollout_instances}"

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=self.collate_fn,
            sampler=self.train_sampler,
        )
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        
    def build_val_dataloader(self) -> None:
        """Build the validation dataloader.
        
        This method creates a StatefulDataLoader for the validation dataset.
        The batch size is determined by `val_batch_size` in the configuration, use the length of the validation dataset as fallback.
        """
        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        print(f"Size of validation dataloader: {len(self.val_dataloader)}")

    def _create_dataloader(self):
        """Create the train and validation dataloaders.
        
        This method initializes the train and validation datasets, samplers, and dataloaders.
        It also calculates the total number of training steps based on the length of the train dataloader and the total epochs.
        If `total_training_steps` is specified in the configuration, it overrides the calculated value.
        """
        self.build_train_and_val_dataset()
        self.build_train_sampler()
        self.build_train_dataloader()
        self.build_val_dataloader()
        
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = min(total_training_steps, self.config.trainer.total_training_steps)
        self.total_training_steps = total_training_steps

    def get_total_training_steps(self):
        """Get the total number of training steps."""
        assert self.total_training_steps is not None, "Total training steps are not set. Call _create_dataloader() first."

        return self.total_training_steps
    
    # ------- Dataloader Management Methods -------
    def save_train_dataloader(self, dataloader_local_path: str) -> None:
        """Save the dataloader to a local path for future resume."""
        assert self.train_dataloader is not None, "Train dataloader is not built yet. Call build_train_dataloader() first."

        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)
        psrl_logger.info(f"Train dataloader saved to {dataloader_local_path}")

    def load_train_dataloader(self, dataloader_local_path: str) -> None:
        """Load the dataloader from a local path."""
        assert self.train_dataloader is not None, "Train dataloader is not built yet. Call build_train_dataloader() first."

        dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
        self.train_dataloader.load_state_dict(dataloader_state_dict)
        psrl_logger.info(f"Train dataloader loaded from {dataloader_local_path}")

    # ------- Data Retrieval Methods -------
    def get_train_next(self):
        """Get the next batch of training data.
        
        This method handles the case where the dataloader iterator is exhausted.
        It will reset the iterator and return the next batch of data.
        
        Returns:
            The next batch of training data.
        
        Raises:
            StopIteration: If all epochs are finished.
        """
        # Initialize the train dataloader iterator if it is None
        if self.train_dataloader_iter is None:
            self.train_dataloader_iter = iter(self.train_dataloader)

        epoch = 0
        try:
            data = next(self.train_dataloader_iter)
        except StopIteration:
            psrl_logger.debug("Train dataloader iterator exhausted.")
            epoch += 1
            if epoch == self.config.trainer.total_epochs:
                psrl_logger.info("All training epochs completed, stopping data processing.")
                raise
            self.train_dataloader_iter = iter(self.train_dataloader)
            data = next(self.train_dataloader_iter)
        return data
    
    def get_val_next(self):
        """Get the next batch of validation data.
        
        This method handles the case where the validation dataloader iterator is exhausted.
        
        It will raise a StopIteration exception if the validation dataloader is exhausted,
        which can be caught by the main controller to handle the end of validation data.
        
        Returns:
            The next batch of validation data.

        Raises:
            StopIteration: If the validation dataloader iterator is exhausted.
        """
        # Initialize the validation dataloader iterator if it is None
        if self.val_dataloader_iter is None:
            self.val_dataloader_iter = iter(self.val_dataloader)

        try:
            data = next(self.val_dataloader_iter)
        except StopIteration:
            psrl_logger.info("Validation dataloader iterator exhausted.")
            self.val_dataloader_iter = iter(self.val_dataloader)
            raise
        return data
    
    def get_train_len(self):
        return len(self.train_dataloader)
    
    def get_val_len(self):
        return len(self.val_dataloader)

    def get_single_controller_batch(self, dataset_type: DatasetType):
        """
        Get a single batch from the dataset for single controller training.
        
        Args:
            dataset_type (DatasetType): The type of dataset to get the batch from (train, val, test).
        
        Returns:
            DataProto: A single batch from the dataset.
        
        Raises:
            StopIteration: If there are no more batches in the dataset.
        """
        if dataset_type == DatasetType.train:
            batch = self.get_train_next()
        elif dataset_type == DatasetType.val:
            batch = self.get_val_next()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        return batch

    # ------- Streaming Data Processing Methods -------
    def start_busy_loop(self):
        """Initialize the thread for data processing."""
        if self.data_process_thread is not None:
            return
            
        self.data_process_thread = threading.Thread(
            target=self._process_data,
            name="data_process_thread",
            daemon=True,
        )
        self.data_process_thread.start()
    
    def stop_busy_loop(self):
        """Stop the data processing thread."""
        if self.data_process_thread is not None:
            self.stop_data_process = True
            self.data_process_thread.join()
            self.data_process_thread = None
    
    def _process_data(self):
        """
        Process the training data in a busy loop.
        
        This method continuously fetches batches from the training dataloader,
        processes them, and puts them into the data queue for further processing.
        
        The method will run until all epochs are processed or a StopIteration is raised.
        After processing, it signals the end of data processing by putting None into the data queue.
        
        Note: This method is designed to run in a separate thread and is intended to be called
        after initializing the DataProcessor and its dataloaders.
        
        The data queue will hold the processed batches, which can be consumed by the rollout server.
        """
        self.train_dataloader_iter = iter(self.train_dataloader)
        total_epochs = self.config.trainer.total_epochs
        
        # loop until all epochs are processed
        while not self.stop_data_process:
            try:
                batch_dict = next(self.train_dataloader_iter)
                batch_size = len(batch_dict[list(batch_dict.keys())[0]])
                sample_ids = [self._train_sample_idx + i for i in range(batch_size)]
                self._train_sample_idx += batch_size

                # For Group Sampling, we use `parent_id` to indicate the shared prompt.
                if self.rollout_n > 1:
                    batch_dict['parent_id'] = np.array(sample_ids)
                else:
                    batch_dict['uid'] = np.array(sample_ids)

                batch_dict = DataProto.from_single_dict(batch_dict)
                
                # Pop the keys that are needed for generation to form the generation batch.
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                meta_info_keys_to_pop = []
                if "multi_modal_inputs" in batch_dict.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
                if "raw_prompt" in batch_dict.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch_dict.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if self.rollout_n > 1:
                    non_tensor_batch_keys_to_pop.append("parent_id")
                else:
                    non_tensor_batch_keys_to_pop.append("uid")
                if "do_sample" in batch_dict.meta_info:
                    meta_info_keys_to_pop.append("do_sample")

                gen_batch = batch_dict.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    meta_info_keys=meta_info_keys_to_pop,
                )
                
                # Store the other batch fields in the request buffer of the ps manager.
                # They will be merged with the reward data.
                ray.get(self.ps_manager_handle.add_request_data_to_buffer.remote(
                    {sample_ids[i]: batch_dict[i:i+1] for i in range(batch_size)}
                ))
                
                # We manually repeat prompts in the generation batch for Group Sampling.
                # Requests in the batch are unique during generation and synchronized through parent tracker.
                psrl_logger.info(f"Generating {batch_size} requests with rollout n {self.rollout_n}")
                if self.rollout_n > 1:
                    gen_batch = gen_batch.repeat(repeat_times=self.rollout_n, interleave=True)
                    uid_list = []
                    for i in range(batch_size):
                        for j in range(self.rollout_n):
                            child_id = sample_ids[i] * self.rollout_n + j
                            uid_list.append(child_id)
                    gen_batch.non_tensor_batch["uid"] = np.array(uid_list)
                
                # Record the request status in the request status manager and put the batch into the data queue.
                if self.process_mode == "stream":
                    # Put group-level requests to data queue
                    parent_ids = gen_batch.non_tensor_batch.get("parent_id", None)
                    assert parent_ids is not None, "parent_id must be present in non_tensor_batch for stream mode with rollout_n > 1"
                    unique_parent_ids = np.unique(parent_ids)
                    for i in unique_parent_ids:
                        sample_idx = i * self.rollout_n
                        ray.get(self.ps_manager_handle.add_request.remote(
                            gen_batch.non_tensor_batch["uid"][sample_idx: (sample_idx + self.rollout_n)].tolist(),
                        ))
                        self.data_queue.put(gen_batch[sample_idx: (sample_idx + self.rollout_n)])
                else:
                    for uid in gen_batch.non_tensor_batch["uid"]:
                        ray.get(self.ps_manager_handle.add_request.remote(uid))
                    self.data_queue.put(gen_batch)
                
                self.global_steps += 1
                if self.total_training_steps is not None and self.global_steps >= self.total_training_steps:
                    self.stop_data_process = True

            except StopIteration:
                curr_epoch = self.global_steps // len(self.train_dataloader)
                psrl_logger.debug(f"StopIteration encountered, current epoch: {curr_epoch}/{total_epochs}")
                if curr_epoch >= total_epochs:
                    psrl_logger.info("All training epochs completed, stopping data processing.")
                    self.stop_data_process = True
                else:
                    psrl_logger.debug("Reinitializing train_dataloader_iter for next epoch")
                    self.train_dataloader_iter = iter(self.train_dataloader)
        
        # Signal end of data processing
        psrl_logger.info("Data processing stopped, sending shutdown signal.")
        self.data_queue.put(None)
