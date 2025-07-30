import os
import numpy as np
import threading
import logging
from typing import Dict, Tuple
from dataclasses import dataclass

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

import ray
from ray.util.queue import Queue as RayQueue

from verl import DataProto

from psrl.utils.dataset.utils import create_rl_dataset, create_rl_sampler
from psrl.utils.logger import log_dual_events, EventType, DualOutputHandler

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

@dataclass
class DatasetType:
    """DataType for the dataset."""
    unknown: str = "unknown"
    train: str = "train"
    val: str = "val"
    test: str = "test"

@ray.remote
class DataProcessor:
    def __init__(
        self,
        config,
        tokenizer,
        processor,
        ps_handle,
        request_status_manager,
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
            ps_handle: Parameter Server handle for communication with the PS worker.
            request_status_manager: Manager for tracking request statuses. (Ray actor handle)
            collate_fn: Function to collate data samples into batches.
            process_mode: Mode of processing data, either "stream" or "batch".
        """

        self.config = config
        
        # Dataset and dataloader attributes
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader_iter = None
        self.val_dataloader_iter = None
        self._train_sample_idx = 0
        self._train_batch_idx = 0 # TODO: will be deprecated in the future
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import \
                collate_fn as default_collate_fn

            self.collate_fn = default_collate_fn
        else:
            self.collate_fn = collate_fn
        
        # Communication handles
        self.ps_handle = ps_handle
        self.request_status_manager = request_status_manager
        
        self.process_mode = process_mode
        if self.config.psrl.rollout_test.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_test.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_test.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, \
            f"Rollout n {self.rollout_n} must be greater than or equal to alg_rollout_n {self.alg_rollout_n}."

        # Data queue is the communication handle between the data processor and the rollout server.
        # It holds the data batches that are ready for processing.
        # The size of the queue is determined by the batch size and the process mode.
        # If process_mode is "stream", it will hold multiple requests for streaming processing.
        # If process_mode is "batch", it will hold a single batch for batch processing.
        self.data_queue_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size) * self.rollout_n if self.process_mode == "stream" else 1
        self.data_queue = RayQueue(maxsize=self.data_queue_size)
        psrl_logger.debug("Created data_queue with maxsize=%d", self.data_queue_size)
        
        # Rollout queue is the communication handle between the rollout workers and the data processor (reward module).
        # It holds the rollout data that is ready for reward computation.
        # The size of the queue is the same as the data queue size.
        self.rollout_queue = RayQueue(maxsize=self.data_queue_size)
        psrl_logger.debug("Created rollout_queue with maxsize=%d", self.data_queue_size)
        
        # Replay buffer is used to store the interrupted requests that need to be replayed.
        # It is a queue that holds the requests that are not yet finished.
        # The size of the replay buffer is not limited, it will grow as needed.
        self.replay_buffer = RayQueue()
        psrl_logger.debug("Created replay_buffer with unlimited size")
        
        # Threads for data processing
        self.data_preprocess_thread = None
        
        # Build logger
        self.log_prefix = "DataProcessor"
        psrl_logger.addHandler(DualOutputHandler(self.log_prefix))
        psrl_logger.info(f"DataProcessor initialized with process_mode={self.process_mode}, data_queue_size={self.data_queue_size}")
        
        # Create the initial datasets and dataloaders
        self._create_dataloader()

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

        if self.config.psrl.rollout_test.redundant_rollout.enable:
            batch_size = self.config.psrl.rollout_test.redundant_rollout.redundant_global_batch_size
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
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps

    def save_train_dataloader(self, dataloader_local_path: str) -> None:
        """Save the dataloader to a local path for future resume."""
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
            print("Train dataloader iterator exhausted.")
            epoch += 1
            if epoch == self.config.trainer.total_epochs:
                print("All epoches finished.")
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
            print("Validation dataloader iterator exhausted.")
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
        psrl_logger.debug("Getting single controller batch for dataset_type: %s", dataset_type)
        
        if dataset_type == DatasetType.train:
            batch = self.get_train_next()
        elif dataset_type == DatasetType.val:
            batch = self.get_val_next()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
        psrl_logger.debug("Got batch with size: %d", len(batch))
        return batch
    
    def get_data_queue(self) -> RayQueue:
        """Get the data queue."""
        return self.data_queue

    def get_rollout_queue(self) -> RayQueue:
        """Get the rollout queue."""
        return self.rollout_queue

    def get_replay_buffer(self) -> RayQueue:
        """Get the replay buffer."""
        return self.replay_buffer
    
    def get_total_training_steps(self):
        """Get the total number of training steps."""
        return self.total_training_steps
    
    def initialize_data_preprocess(self):
        """Initialize the thread for data preprocessing."""
        if self.data_preprocess_thread is not None:
            psrl_logger.debug("Data preprocessing thread already exists, skipping initialization")
            return
            
        self.data_preprocess_thread = threading.Thread(
            target=self.preprocess_data,
            name="data_preprocess_thread",
            daemon=True,
        )
        self.data_preprocess_thread.start()
    
    def preprocess_data(self):
        """
        Preprocess the training data in a busy loop.
        
        This method continuously fetches batches from the training dataloader,
        processes them, and puts them into the data queue for further processing.
        
        The method will run until all epochs are processed or a StopIteration is raised.
        After processing, it signals the end of data processing by putting None into the data queue.
        
        Note: This method is designed to run in a separate thread and is intended to be called
        after initializing the DataProcessor and its dataloaders.
        
        The data queue will hold the processed batches, which can be consumed by the rollout server.
        """
        psrl_logger.debug("Starting preprocess_data method in DataProcessor")
        self.train_dataloader_iter = iter(self.train_dataloader)
        total_epochs = self.config.trainer.total_epochs
        psrl_logger.debug(f"Total epochs to process: {total_epochs}")
        
        # loop until all epochs are processed
        while True:
            try:
                psrl_logger.debug(f"Fetching next batch from train_dataloader_iter, current batch_idx: {self._train_batch_idx}")
                batch_dict = next(self.train_dataloader_iter)
                self._train_batch_idx += 1
                batch_size = len(batch_dict[list(batch_dict.keys())[0]])
                sample_ids = [self._train_sample_idx + i for i in range(batch_size)]
                self._train_sample_idx += batch_size
                psrl_logger.debug(f"Got batch with {batch_size} samples, sample_ids: {sample_ids[:5]}{'...' if len(sample_ids) > 5 else ''}")

                # For Group Sampling, we use `parent_id` to indicate the shared prompt.
                if self.rollout_n > 1:
                    batch_dict['parent_id'] = np.array(sample_ids)
                    psrl_logger.debug(f"Using Group Sampling with rollout_n={self.rollout_n}, added parent_id")
                else:
                    batch_dict['uid'] = np.array(sample_ids)
                    psrl_logger.debug("Added uid for standard sampling")
                # Record partial response ids for resume
                batch_dict['raw_response_ids'] = np.fromiter(([] for _ in range(batch_size)), dtype=object)
                
                meta_info = {
                    'batch_idx': self._train_batch_idx, # TODO: used for logging, not necessary
                }
                
                batch_dict = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
                
                # Pop the keys that are needed for generation to form the generation batch.
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "raw_response_ids"]
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

                psrl_logger.debug(f"Keys to pop for gen_batch - batch: {batch_keys_to_pop}, non_tensor: {non_tensor_batch_keys_to_pop}, meta_info: {meta_info_keys_to_pop}")
                gen_batch = batch_dict.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    meta_info_keys=meta_info_keys_to_pop,
                )
                psrl_logger.debug(f"Created gen_batch with size {len(gen_batch)}")
                
                # Store the other batch data in the request buffer of the request status manager.
                # They will be merged with the reward data.
                psrl_logger.debug(f"Adding {batch_size} requests to buffer in request_status_manager")
                ray.get(self.request_status_manager.add_request_data_to_buffer.remote(
                    {sample_ids[i]: batch_dict[i:i+1] for i in range(batch_size)}
                ))
                psrl_logger.debug("Successfully added requests to buffer")
                
                # We manually repeat prompts in the generation batch for Group Sampling.
                # Requests in the batch are unique during generation and synchronized through parent tracker.
                if self.rollout_n > 1:
                    psrl_logger.debug(f"Repeating gen_batch for Group Sampling with rollout_n={self.rollout_n}")
                    gen_batch = gen_batch.repeat(repeat_times=self.rollout_n, interleave=True)
                    uid_list = []
                    for i in range(batch_size):
                        for j in range(self.rollout_n):
                            child_id = sample_ids[i] * self.rollout_n + j
                            uid_list.append(child_id)
                    gen_batch.non_tensor_batch["uid"] = np.array(uid_list)
                    psrl_logger.debug(f"Created {len(uid_list)} child UIDs for Group Sampling, first few: {uid_list[:5]}{'...' if len(uid_list) > 5 else ''}")
                
                # Record the request status in the request status manager and put the batch into the data queue.
                if self.process_mode == "stream":
                    psrl_logger.debug(f"Process mode: stream, adding {len(gen_batch)} individual requests to data_queue")
                    batch_size = len(gen_batch)
                    for i in range(batch_size):
                        ray.get(self.request_status_manager.add_request.remote(
                            gen_batch.non_tensor_batch["uid"][i],
                        ))
                        self.data_queue.put(gen_batch[i:i+1])
                    psrl_logger.debug(f"Added {batch_size} individual requests to data_queue")
                else:
                    psrl_logger.debug(f"Process mode: batch, adding {len(gen_batch.non_tensor_batch['uid'])} UIDs as a batch to data_queue")
                    for uid in gen_batch.non_tensor_batch["uid"]:
                        ray.get(self.request_status_manager.add_request.remote(uid))
                    self.data_queue.put(gen_batch)
                    psrl_logger.debug(f"Added batch with {len(gen_batch)} samples to data_queue")
                
                if self._train_batch_idx > self.total_training_steps:
                    psrl_logger.debug(f"Reached total_training_steps: {self.total_training_steps}, breaking loop")
                    break

            except StopIteration:
                curr_epoch = self._train_batch_idx // len(self.train_dataloader)
                psrl_logger.debug(f"StopIteration encountered, current epoch: {curr_epoch}/{total_epochs}")
                if curr_epoch >= total_epochs:
                    psrl_logger.info("All training epochs completed.")
                    break
                else:
                    psrl_logger.debug("Reinitializing train_dataloader_iter for next epoch")
                    self.train_dataloader_iter = iter(self.train_dataloader)
        
        # Signal end of data processing
        psrl_logger.info("Data processing completed, sending shutdown signal to reward computation thread.")
        self.data_queue.put(None)
