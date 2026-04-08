import logging
import os
import sys
import threading
from dataclasses import dataclass

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from verl import DataProto
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

from psrl.utils.dataset.utils import create_multi_rl_datasets
from psrl.utils.logger import DualOutputHandler, log_data_protocol

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
        collate_fn=None,
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
        """

        self.config = config

        # Dataset and dataloader attributes
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader_iters = None
        self.val_dataloader_iters = None
        # train_datasets_ratios is set here for multi-dataset mode;
        # in legacy mode it will be overridden to [1.0] in build_train_and_val_dataset().
        if self.config.data.get("use_multi_dataset", True):
            self.train_datasets_ratios = self.config.data.train_datasets_ratios
        else:
            self.train_datasets_ratios = [1.0]
        self._validate_data_config()
        self._val_dataloader_idx = 0  # Index for round-robin selection of validation dataloaders

        self.global_steps = 0
        self._train_sample_idx = 0
        self._val_sample_idx = 0

        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            self.collate_fn = default_collate_fn
        else:
            self.collate_fn = collate_fn

        # Communication handles
        self.ps_manager_handle = ps_manager_handle
        self.reward_manager_handle = None  # Will be set by the ray trainer

        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, (
            f"Rollout n {self.rollout_n} must be greater than or equal to alg_rollout_n {self.alg_rollout_n}."
        )
        self.val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n

        # Use conservative bounds: divide available space by 2 for train and eval each
        # This ensures: MAX_TRAIN_ID * rollout_n < sys.maxsize // 2
        # MAX_VAL_ID * val_rollout_n < sys.maxsize // 2
        self.MAX_TRAIN_ID = sys.maxsize // (2 * self.rollout_n)
        self.MAX_VAL_ID = sys.maxsize // (2 * self.val_rollout_n)

        # Threads for data processing
        self.data_process_thread = None
        self.stop_data_process = False

        # Build logger
        self.log_prefix = "DataProcessor"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

        # Create the initial datasets and dataloaders
        self.total_training_steps = None
        self._create_dataloader()

    def _validate_data_config(self) -> None:
        """Validate the data configuration for consistency.

        In multi-dataset mode (use_multi_dataset=True), train_datas and
        train_datasets_ratios must both be set.

        In legacy mode (use_multi_dataset=False), train_files must be set.
        If train_datas is also set, a warning is emitted (it will be ignored).
        """
        if self.config.data.get("use_multi_dataset", True):
            if not self.config.data.get("train_datas"):
                raise ValueError("use_multi_dataset=True requires 'train_datas' to be set in data config.")
            if not self.config.data.get("train_datasets_ratios"):
                raise ValueError("use_multi_dataset=True requires 'train_datasets_ratios' to be set in data config.")
        else:
            if not self.config.data.get("train_files"):
                raise ValueError("use_multi_dataset=False requires 'train_files' to be set in data config.")
            if self.config.data.get("train_datas"):
                psrl_logger.warning(
                    "use_multi_dataset=False but 'train_datas' is set in data config — it will be ignored."
                )

    def set_reward_manager(self, reward_manager_handle: ray.actor.ActorHandle):
        self.reward_manager_handle = reward_manager_handle

    def set_agent_loop_manager(self, agent_loop_manager_handle: ray.actor.ActorHandle):
        self.agent_loop_manager_handle = agent_loop_manager_handle

    # ------- Dataset and Dataloader Building Methods -------
    def build_train_and_val_dataset(self) -> None:
        """Build the training and validation datasets.

        Behavior depends on ``config.data.use_multi_dataset``:

        * ``True`` (default): uses PSRL multi-dataset path.  Each entry in
          ``config.data.train_datas`` / ``config.data.val_datas`` is turned into
          a separate ``RLHFDataset``; oversampling is applied so that all datasets
          contribute the same number of batches per epoch.

        * ``False``: falls back to the veRL-native single-dataset path.
          ``config.data.train_files`` / ``config.data.val_files`` are used; no
          oversampling is performed, and ``train_datasets_ratios`` is forced to
          ``[1.0]`` so that the rest of the dataloader pipeline sees a
          single-element list transparently.
        """
        if self.config.data.get("use_multi_dataset", True):
            # ── Multi-dataset path (PSRL) ──────────────────────────────────────
            self.train_datasets = create_multi_rl_datasets(
                self.config.data.train_datas,
                self.tokenizer,
                self.processor,
            )

            dataset_sizes = [len(dataset) for dataset in self.train_datasets]
            rough_batch_sizes = [
                int(self.config.data.train_batch_size * ratio) for ratio in self.train_datasets_ratios
            ]
            rough_num_batches = [
                dataset_size // rough_batch_size
                for dataset_size, rough_batch_size in zip(dataset_sizes, rough_batch_sizes)
            ]
            max_num_batches = max(rough_num_batches) if rough_num_batches else 0
            oversample_ratios = [
                max_num_batches / rough_num_batches[i] if rough_num_batches[i] > 0 else 1.0
                for i in range(len(self.train_datasets))
            ]
            for dataset, oversample_ratio in zip(self.train_datasets, oversample_ratios):
                dataset.over_sample_dataset(oversample_ratio)

            self.val_datasets = create_multi_rl_datasets(
                self.config.data.val_datas,
                self.tokenizer,
                self.processor,
            )
        else:
            # ── Legacy path (veRL-compatible) ──────────────────────────────────
            # Single dataset wrapped in a list so the rest of the pipeline
            # (build_train_dataloader, _concat_and_shuffle_batch_dicts, etc.)
            # works transparently without any further changes.
            self.train_datasets = [
                create_rl_dataset(
                    data_paths=self.config.data.train_files,
                    data_config=self.config.data,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    max_samples=self.config.data.get("train_max_samples", -1),
                )
            ]
            self.val_datasets = [
                create_rl_dataset(
                    data_paths=self.config.data.val_files,
                    data_config=self.config.data,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    max_samples=self.config.data.get("val_max_samples", -1),
                )
            ]
            # Force single-element ratio so build_train_dataloader works unchanged.
            self.train_datasets_ratios = [1.0]

    def build_train_sampler(self) -> None:
        """Build the training sampler."""
        assert self.train_datasets is not None, (
            "Train datasets are not built yet. Call build_train_and_val_dataset() first."
        )

        self.train_samplers = [create_rl_sampler(self.config.data, dataset) for dataset in self.train_datasets]

    def build_train_dataloader(self) -> None:
        """Build the training dataloader.

        This method creates a StatefulDataLoader for the training dataset.
        Note that the batch size is determined by `gen_batch_size` in the configuration,
        use `train_batch_size` as fallback.
        """
        assert self.train_datasets is not None, (
            "Train datasets are not built yet. Call build_train_and_val_dataset() first."
        )
        assert self.train_samplers is not None, "Train samplers are not built yet. Call build_train_sampler() first."
        assert len(self.train_datasets) == len(self.train_samplers), (
            "The number of train datasets and train samplers must be the same."
        )
        assert len(self.train_datasets_ratios) == len(self.train_datasets), (
            "The number of train datasets and train datasets ratios must be the same."
        )

        if self.config.psrl.redundant_rollout.enable:
            total_batch_size = self.config.psrl.redundant_rollout.redundant_global_batch_size
        else:
            total_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)

        batch_sizes = [int(total_batch_size * ratio) for ratio in self.train_datasets_ratios]
        batch_sizes[-1] = total_batch_size - sum(batch_sizes[:-1])
        psrl_logger.info(f"---Train Datasets--- batch sizes: {batch_sizes}, total batch size: {total_batch_size}")

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloaders = [
            StatefulDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
                collate_fn=self.collate_fn,
                sampler=sampler,
            )
            for dataset, batch_size, sampler in zip(self.train_datasets, batch_sizes, self.train_samplers)
        ]

        self.train_dataloader_sizes = [len(dataloader) for dataloader in self.train_dataloaders]
        psrl_logger.info(f"Size of train dataloader: {self.train_dataloader_sizes}")

    def build_val_dataloader(self) -> None:
        """Build the validation dataloader.

        This method creates a StatefulDataLoader for the validation dataset.
        The batch size is determined by `val_batch_size` in the configuration,
        use the length of the validation dataset as fallback.
        """
        assert self.val_datasets is not None, (
            "Validation datasets are not built yet. Call build_train_and_val_dataset() first."
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_sizes = [len(dataset) for dataset in self.val_datasets]
        else:
            # Use the same batch size for all validation datasets
            val_batch_sizes = [val_batch_size] * len(self.val_datasets)
        psrl_logger.info(f"---Val Datasets--- batch sizes: {val_batch_sizes}")

        num_workers = self.config.data["dataloader_num_workers"]

        self.val_dataloaders = [
            StatefulDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=self.collate_fn,
            )
            for dataset, batch_size in zip(self.val_datasets, val_batch_sizes)
        ]

        self.val_dataloader_sizes = [len(dataloader) for dataloader in self.val_dataloaders]
        psrl_logger.info(f"Size of validation dataloaders: {self.val_dataloader_sizes}")

    def _create_dataloader(self):
        """Create the train and validation dataloaders.

        This method initializes the train and validation datasets, samplers, and dataloaders.
        It also calculates the total number of training steps
        based on the length of the train dataloader and the total epochs.
        If `total_training_steps` is specified in the configuration, it overrides the calculated value.
        """

        # AGENT(VERL): this logic is mapped to `_create_dataloader(...)` in `verl/trainer/ppo/ray_trainer.py`

        self.build_train_and_val_dataset()
        self.build_train_sampler()
        self.build_train_dataloader()
        self.build_val_dataloader()

        total_training_steps = min(self.train_dataloader_sizes) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = min(total_training_steps, self.config.trainer.total_training_steps)
        self.total_training_steps = total_training_steps

    def get_val_data_size(self):
        """Get the size of the validation dataset."""
        assert self.val_dataloaders is not None, (
            "Validation dataloaders are not built yet. Call build_val_dataloader() first."
        )
        return sum([len(dataloader.dataset) for dataloader in self.val_dataloaders])

    def get_val_batch_num(self):
        """Get the number of batches in the validation dataloader."""
        assert self.val_dataloaders is not None, (
            "Validation dataloaders are not built yet. Call build_val_dataloader() first."
        )
        return sum([len(dataloader) for dataloader in self.val_dataloaders])

    def get_total_training_steps(self):
        """Get the total number of training steps."""
        assert self.total_training_steps is not None, (
            "Total training steps are not set. Call _create_dataloader() first."
        )

        return self.total_training_steps

    # ------- Dataloader Management Methods -------
    def save_train_dataloader(self, dataloader_local_paths: list[str] | str) -> None:
        """Save the dataloader to a local path for future resume."""
        assert self.train_dataloaders is not None, (
            "Train dataloader is not built yet. Call build_train_dataloader() first."
        )
        if not isinstance(dataloader_local_paths, list):
            dataloader_local_paths = [dataloader_local_paths] * len(self.train_dataloaders)
        assert len(dataloader_local_paths) == len(self.train_dataloaders), (
            "The number of dataloader paths must match the number of train dataloaders."
        )

        # AGENT(VERL): this logic is mapped to `_save_checkpoint(...)` in `verl/trainer/ppo/ray_trainer.py`
        # you can find it by searching for `# save dataloader` in that file.

        for i, dataloader in enumerate(self.train_dataloaders):
            torch.save(dataloader.state_dict(), dataloader_local_paths[i])
        psrl_logger.info(f"Train dataloader saved to {dataloader_local_paths}")

    def load_train_dataloader(self, dataloader_local_paths: list[str] | str) -> None:
        """Load the dataloader from a local path."""
        assert self.train_dataloaders is not None, (
            "Train dataloader is not built yet. Call build_train_dataloader() first."
        )
        if not isinstance(dataloader_local_paths, list):
            dataloader_local_paths = [dataloader_local_paths] * len(self.train_dataloaders)
        assert len(dataloader_local_paths) == len(self.train_dataloaders), (
            "The number of dataloader paths must match the number of train dataloaders."
        )

        # AGENT(VERL): this logic is mapped to `_load_checkpoint(...)` in `verl/trainer/ppo/ray_trainer.py`
        # you can find it by searching for `# load dataloader` in that file
        steps_per_epoch = min(self.train_dataloader_sizes)
        at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
        if at_epoch_boundary:
            psrl_logger.info(
                f"Skipping dataloader state restore: global_steps={self.global_steps} "
                f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                f"The saved state marks the dataloader as exhausted. "
                f"Next epoch will iterate from scratch."
            )
        else:
            for dataloader, dataloader_local_path in zip(self.train_dataloaders, dataloader_local_paths):
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                dataloader.load_state_dict(dataloader_state_dict)
            psrl_logger.info(f"Train dataloader loaded from {dataloader_local_paths}")

    # ------- Data Retrieval Methods -------
    def get_train_next(self) -> dict:
        """Get the next merged batch from all training dataloaders.

        Fetches one batch from each training dataloader, concatenates them with
        segment-level shuffling (preserving per-dataset row order), and returns
        the combined dict.

        Lazy-initializes the dataloader iterators on first call.

        Returns:
            dict: A merged batch dict from all training datasets.

        Raises:
            StopIteration: When any training dataloader is exhausted (epoch boundary).
                All iterators are reset before re-raising so the next call starts
                a fresh epoch.
        """
        if self.train_dataloader_iters is None:
            self.train_dataloader_iters = [iter(dl) for dl in self.train_dataloaders]

        try:
            batch_dicts = [next(it) for it in self.train_dataloader_iters]
        except StopIteration:
            # Reset all iterators so the caller can start the next epoch cleanly.
            self.train_dataloader_iters = [iter(dl) for dl in self.train_dataloaders]
            raise

        return self._concat_and_shuffle_batch_dicts(batch_dicts)

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
        if self.val_dataloader_iters is None:
            self.val_dataloader_iters = [iter(dataloader) for dataloader in self.val_dataloaders]
            self._val_dataloader_idx = 0

        # Sequentially get data from each dataloader, starting from the first one
        # Continue with the current dataloader until it's exhausted, then move to the next
        num_dataloaders = len(self.val_dataloader_iters)

        while self._val_dataloader_idx < num_dataloaders:
            i = self._val_dataloader_idx
            dataloader_iter = self.val_dataloader_iters[i]
            try:
                data = next(dataloader_iter)
                return data
            except StopIteration:
                psrl_logger.info(f"Validation dataloader {i} iterator exhausted.")
                # Reset the exhausted iterator and move to next dataloader
                self.val_dataloader_iters[i] = iter(self.val_dataloaders[i])
                self._val_dataloader_idx += 1
                continue

        # If we reach here, all dataloaders have been exhausted
        # Reset index for next epoch
        self._val_dataloader_idx = 0
        raise StopIteration("All validation dataloader iterators exhausted.")

    def get_val_sample_ids(self, batch_size: int) -> list:
        """
        Generate sample IDs for validation data with cyclic wrapping.

        This method generates unique sample IDs for validation data that are guaranteed
        not to conflict with training sample IDs by using a separate ID namespace.
        The IDs cycle within [MAX_TRAIN_ID, MAX_TRAIN_ID + MAX_VAL_ID) to avoid overflow.

        Args:
            batch_size (int): The number of sample IDs to generate.

        Returns:
            list: A list of sample IDs for the validation batch.
        """
        sample_ids = []
        for _ in range(batch_size):
            # Cycle within the validation namespace: [MAX_TRAIN_ID, MAX_TRAIN_ID + MAX_VAL_ID)
            val_offset = self._val_sample_idx % self.MAX_VAL_ID
            sample_id = self.MAX_TRAIN_ID + val_offset
            sample_ids.append(sample_id)
            self._val_sample_idx += 1
        return sample_ids

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

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # Keys that must be preserved so the reward manager and agent loop can consume them.
        # - data_source, reward_model, extra_info, uid: always required by reward manager.
        # - reward_model_dicts: new RLHFDataset field replacing the old reward_fn_key / reward_model_dict.
        # - tools_kwargs: per-sample tool execution kwargs (new RLHFDataset field).
        # - interaction_kwargs: per-sample interaction kwargs (new RLHFDataset field).
        reward_keys = (
            set(
                {
                    "data_source",
                    "reward_model",
                    "extra_info",
                    "uid",
                    "reward_model_dicts",
                    "tools_kwargs",
                    "interaction_kwargs",
                }
            )
            & batch.non_tensor_batch.keys()
        )

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    @staticmethod
    def _concat_and_shuffle_batch_dicts(batch_dicts: list[dict]) -> dict:
        """Concatenate batch dicts and shuffle at the dataset-segment level.

        Each dataset's rows are concatenated in a random segment order, but the
        relative row order *within* each dataset's segment is preserved.

        Args:
            batch_dicts: One dict per dataset, each mapping feature names to
                torch.Tensor or np.ndarray of shape (batch_size, *dims).

        Returns:
            A single dict with all features concatenated along dim 0, with
            segments arranged in a random order.

        Raises:
            ValueError: If batch_dicts is empty or any individual dict is empty.
            TypeError: If a value is neither torch.Tensor nor np.ndarray.
        """
        if not batch_dicts:
            raise ValueError("batch_dicts must not be empty.")

        num_datasets = len(batch_dicts)

        # Validate and collect per-dataset sizes
        for batch_dict in batch_dicts:
            if not batch_dict:
                raise ValueError("Batch dict must not be empty.")

        # Shuffle segment order (dataset-level permutation)
        segment_order = torch.randperm(num_datasets).tolist() if num_datasets > 1 else [0]
        shuffled_dicts = [batch_dicts[i] for i in segment_order]

        # Collect the union of all keys, using first occurrence as the type sample
        key_sample: dict[str, torch.Tensor | np.ndarray] = {}
        key_order: list[str] = []
        batch_sizes: list[int] = []

        for batch_dict in shuffled_dicts:
            first_value = next(iter(batch_dict.values()))
            batch_sizes.append(len(first_value))
            for key, value in batch_dict.items():
                if key not in key_sample:
                    key_sample[key] = value
                    key_order.append(key)

        # Concatenate each key across shuffled segments, filling missing keys with zeros/None
        concatenated: dict = {}
        for key in key_order:
            sample_value = key_sample[key]
            values = [
                batch_dict[key]
                if key in batch_dict
                else DataProcessor._create_placeholder_like(sample_value, batch_size)
                for batch_dict, batch_size in zip(shuffled_dicts, batch_sizes)
            ]
            if isinstance(sample_value, torch.Tensor):
                concatenated[key] = torch.cat(values, dim=0)
            elif isinstance(sample_value, np.ndarray):
                concatenated[key] = np.concatenate(values, axis=0)
            else:
                raise TypeError(f"Unsupported value type {type(sample_value)} for key '{key}' in batch concatenation.")

        return concatenated

    @staticmethod
    def _create_placeholder_like(sample_value: torch.Tensor | np.ndarray, batch_size: int):
        """Create a placeholder tensor/array matching the sample value's dtype and trailing shape."""
        if isinstance(sample_value, torch.Tensor):
            shape = (batch_size, *sample_value.shape[1:])
            return torch.zeros(shape, dtype=sample_value.dtype)
        if isinstance(sample_value, np.ndarray):
            shape = (batch_size, *sample_value.shape[1:])
            if sample_value.dtype == object:
                placeholder = np.empty(shape, dtype=object)
                placeholder[:] = None
                return placeholder
            return np.zeros(shape, dtype=sample_value.dtype)
        raise TypeError(f"Unsupported value type {type(sample_value)} for placeholder creation.")

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
        assert self.reward_manager_handle is not None, (
            "Reward manager handle is not set. Call `reward_manager_handle()` first."
        )
        self.train_dataloader_iters = [iter(dataloader) for dataloader in self.train_dataloaders]
        total_epochs = self.config.trainer.total_epochs

        # loop until all epochs are processed
        while not self.stop_data_process:
            try:
                # AGENT(VERL): this process is mapped to the data loading and processing logic in
                # `fit(...)` in `verl/trainer/ppo/ray_trainer.py`, you can find it by searching for
                # `batch: DataProto = DataProto.from_single_dict(batch_dict)` in that file.

                batch_dicts = [next(dataloader_iter) for dataloader_iter in self.train_dataloader_iters]
                batch_dict = self._concat_and_shuffle_batch_dicts(batch_dicts)
                batch_size = len(batch_dict[list(batch_dict.keys())[0]])

                # Generate training sample IDs with cyclic wrapping to avoid overflow
                sample_ids = []
                for i in range(batch_size):
                    sample_id = (self._train_sample_idx + i) % self.MAX_TRAIN_ID
                    sample_ids.append(sample_id)
                self._train_sample_idx = (self._train_sample_idx + batch_size) % self.MAX_TRAIN_ID

                # For Group Sampling, we use `parent_id` to indicate the shared prompt.
                if self.rollout_n > 1:
                    batch_dict["parent_id"] = np.array(sample_ids)
                else:
                    batch_dict["uid"] = np.array(sample_ids)

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.gen_actor_rollout_ref.rollout.temperature

                # TODO(linsh): check whether get_gen_batch is needed
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps

                # Store the other batch fields in the request buffer of the reward manager
                # They will be merged with the reward data.
                log_data_protocol(
                    batch,
                    psrl_logger,
                    self.log_prefix + " before adding request data to ps manager",
                    level=logging.DEBUG,
                )
                ray.get(
                    self.reward_manager_handle.add_requests.remote(
                        {sample_ids[i]: batch[i : i + 1] for i in range(batch_size)}
                    )
                )

                # We manually repeat prompts in the generation batch for Group Sampling.
                # Requests in the batch are unique during generation and synchronized through parent tracker.
                psrl_logger.debug(f"Generating {batch_size} requests with rollout n {self.rollout_n}")
                if self.rollout_n > 1:
                    gen_batch = gen_batch.repeat(repeat_times=self.rollout_n, interleave=True)
                    uid_list = []
                    for i in range(batch_size):
                        for j in range(self.rollout_n):
                            child_id = sample_ids[i] * self.rollout_n + j
                            uid_list.append(child_id)
                    gen_batch.non_tensor_batch["uid"] = np.array(uid_list)

                # Record the request status in the request status manager and put the batch into the data queue.
                # Put group-level requests to data queue
                for i in range(batch_size):
                    ray.get(
                        self.ps_manager_handle.add_request.remote(
                            gen_batch.non_tensor_batch["uid"][i * self.rollout_n : (i + 1) * self.rollout_n].tolist(),
                        )
                    )
                    ray.get(
                        self.agent_loop_manager_handle.put_data.remote(
                            gen_batch[i * self.rollout_n : (i + 1) * self.rollout_n]
                        )
                    )

                self.global_steps += 1
                if self.total_training_steps is not None and self.global_steps >= self.total_training_steps:
                    self.stop_data_process = True

            except StopIteration:
                curr_epoch = self.global_steps // min(self.train_dataloader_sizes)
                psrl_logger.debug(f"StopIteration encountered, current epoch: {curr_epoch}/{total_epochs}")
                if curr_epoch >= total_epochs:
                    psrl_logger.info("All training epochs completed, stopping data processing.")
                    self.stop_data_process = True
                else:
                    psrl_logger.debug("Reinitializing train_dataloader_iters for next epoch")
                    self.train_dataloader_iters = [iter(dl) for dl in self.train_dataloaders]
            except Exception as e:
                psrl_logger.error(f"Exception in data processing thread: {e}", exc_info=True)

        # Signal end of data processing
        psrl_logger.info("Data processing stopped, sending shutdown signal.")
        self.agent_loop_manager_handle.put_data.remote(None)
