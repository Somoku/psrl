import os
import numpy as np
import threading
import logging
from typing import Dict, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

import ray
from ray.util.queue import Queue as RayQueue

from verl import DataProto
from verl.utils.import_utils import load_extern_type
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.torch_functional import pad_2d_list_to_length

from psrl.utils.logger import log_dual_events, EventType
from psrl.workers.gen import RolloutCommand, CommandType

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

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
    psrl_logger.info(f"Using dataset class: {dataset_cls.__name__}")

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
class DataProcessor:
    def __init__(
        self,
        config,
        tokenizer,
        processor,
        ps_handle,
        rollout_instances_strategy: Dict[int, Tuple[int, int]],
        collate_fn=None,
        reward_fn=None,
        process_mode="batch",
        use_rm=False,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import \
                collate_fn as default_collate_fn

            self.collate_fn = default_collate_fn
        else:
            self.collate_fn = collate_fn
        
        self.use_rm = use_rm
        self.reward_fn = reward_fn
        
        self.ps_handle = ps_handle
        self.rollout_server_ref = None
        
        if self.config.psrl.rollout_test.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_test.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_test.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, \
            f"Rollout n {self.rollout_n} must be greater than or equal to alg_rollout_n {self.alg_rollout_n}."

        self.process_mode = process_mode
        self.data_queue_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size) * self.rollout_n if self.process_mode == "stream" else 1
        self.data_queue = RayQueue(maxsize=self.data_queue_size)
        self.rollout_queue = RayQueue(maxsize=self.data_queue_size)
        self.replay_buffer = RayQueue(maxsize=self.data_queue_size)
        self.rollout_request_buffer = {}
        self.rollout_parent_counter = {}

        self.rollout_instances_strategy = rollout_instances_strategy # key: rollout instance id, value: tp size
        self.rollout_instance_num = len(rollout_instances_strategy)
        
        self._threads = []
        self._reward_shutdown = False

        self.train_dataloader_iter = None
        self.val_dataloader_iter = None
        
        self._train_batch_idx = 0
        self._train_sample_idx = 0
        
        psrl_logger.info(f"DataProcessor initialized with process_mode={self.process_mode}, data_queue_size={self.data_queue_size}")
        self._create_dataloader()

    def set_rollout_server_ref(self, rollout_server_ref):
        self.rollout_server_ref = rollout_server_ref

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
        assert batch_size % self.config.psrl.deployment.n_rollout_instances == 0, \
            f"Batch size {batch_size} is not divisible by" \
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
        """Create the train and validation dataloaders."""
        self.build_train_and_val_dataset()
        self.build_train_sampler()
        self.build_train_dataloader()
        self.build_val_dataloader()
        
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps

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

    # Get the current batch for the main controller
    # Just directly return the next batch   
    def get_single_controller_batch(self, dataset_type: DatasetType):
        get_next_data_func = self.get_train_next if dataset_type == DatasetType.train else self.get_val_next
        try:
            return get_next_data_func()
        except StopIteration:
            print("get_single_controller_batch() runs into StopIteration")
            raise
    
    def initialize_data_preprocess(self):
        preprocess_thread = threading.Thread(
            target=self.preprocess_data,
            name="data_preprocess_thread",
            daemon=True,
        )
        preprocess_thread.start()
        self._threads.append(preprocess_thread)

    def initialize_reward_computation(self):
        reward_thread = threading.Thread(
            target=self.compute_reward,
            name="compute_reward_thread",
            daemon=True,
        )
        reward_thread.start()
        self._threads.append(reward_thread)
    
    def preprocess_data(self):
        self.train_dataloader_iter = iter(self.train_dataloader)
        total_epochs = self.config.trainer.total_epochs
        
        # loop until all epochs are processed
        while True:
            try:
                batch_dict = next(self.train_dataloader_iter)
                self._train_batch_idx += 1
                batch_size = len(batch_dict[list(batch_dict.keys())[0]])
                sample_ids = [self._train_sample_idx + i for i in range(batch_size)]
                self._train_sample_idx += batch_size
                # sample_ids = [f"b{self._train_batch_idx}_s{i}" for i in range(batch_size)]
                if self.rollout_n > 1:
                    batch_dict['parent_id'] = np.array(sample_ids)
                else:
                    batch_dict['uid'] = np.array(sample_ids)
                meta_info = {
                    'batch_idx': self._train_batch_idx,
                }
                
                batch_dict = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
                
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
                '''
                gen_batch contains:
                - batch: input_ids, attention_mask, position_ids
                - non_tensor_batch: raw_prompt_ids, raw_prompt, multi_modal_data, multi_modal_inputs, tools_kwargs, uid, (parent_id)
                - meta_info: (do_sample)
                '''
                gen_batch = batch_dict.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    meta_info_keys=meta_info_keys_to_pop,
                )
                
                for i in range(batch_size):
                    self.rollout_request_buffer[sample_ids[i]] = batch_dict[i:i+1]
                
                if self.rollout_n > 1:
                    gen_batch = gen_batch.repeat(repeat_times=self.rollout_n, interleave=True)
                    uid_list = []
                    for i in range(batch_size):
                        for j in range(self.rollout_n):
                            # child_id = f"{sample_ids[i]}_r{j}"
                            child_id = sample_ids[i] * self.rollout_n + j
                            uid_list.append(child_id)
                    gen_batch.non_tensor_batch["uid"] = np.array(uid_list)
                
                if self.process_mode == "stream":
                    batch_size = len(gen_batch)
                    for i in range(batch_size):
                        self.data_queue.put(gen_batch[i:i+1])
                else:
                    self.data_queue.put(gen_batch)
                
                if self._train_batch_idx > self.total_training_steps:
                    break

            except StopIteration:
                curr_epoch = self._train_batch_idx // len(self.train_dataloader)
                if curr_epoch >= total_epochs:
                    psrl_logger.info("All training epochs completed.")
                    break
                else:
                    self.train_dataloader_iter = iter(self.train_dataloader)
        # Signal end of data processing
        psrl_logger.info("Data processing completed, sending shutdown signal to reward computation thread.")
        self.data_queue.put(None)
    
    def compute_reward(self):
        '''
        rollout_queue存放的是已经完成的rollout数据，
        需要从中提取出reward计算所需的输入数据，并计算reward。
        计算完成后，将结果存入PS worker，并更新rollout server的状态。
        
        如果rollout server收集满一个batch，或者一个父请求满，则需要
        (1). 舍弃相关的 rollout 请求
        (2). 舍弃相关的 reward 计算
        '''
        assert self.rollout_server_ref is not None, "Rollout server reference is not set. Call set_rollout_server_ref() first."
        future_reward_buffer = [] if self.config.psrl.log_prob.enable_inference_engine_log_prob else None
        
        while not self._reward_shutdown:
            try:
                rollout_data = self.rollout_queue.get(block=False)
                if rollout_data is None:
                    psrl_logger.info("Received shutdown signal in batching thread.")
                    self._reward_shutdown = True
                    continue
                request_ids = rollout_data.non_tensor_batch["uid"]
                
                # 区分已完成和未完成(interrupted)，并区分interrupt和abort
                # Interrupt 部分要放入 replay buffer
                interrupted = rollout_data.non_tensor_batch["interrupted"]
                interrupted_idxs = np.where(interrupted)[0]
                finished_idxs = np.where(~interrupted)[0]
                
                finished_request_ids = rollout_data.non_tensor_batch["uid"][finished_idxs]
                ray.get(self.rollout_server_ref.finish_requests.remote(request_ids))
                
                finished_rollout_data = rollout_data.select_idxs(finished_idxs)
                interrupted_rollout_data = rollout_data.select_idxs(interrupted_idxs)
                if len(interrupted_rollout_data) > 0:
                    is_aborted = ray.get(self.rollout_server_ref.is_aborted.remote(interrupted_rollout_data.non_tensor_batch["uid"]))
                    replay_idxs = np.where(~is_aborted)[0]
                    interrupt_request_ids = interrupted_rollout_data.non_tensor_batch["uid"][replay_idxs]
                    # Mark the interrupted requests as interrupted in the tracker
                    update_version_tag = self.config.psrl.rollout_test.partial_rollout.interrupt_as_prompt
                    ray.get(self.rollout_server_ref.interrupt_requests.remote(interrupt_request_ids, update_version_tag))
                    # Store interrupt data to replay buffer
                    replay_rollout_data = interrupted_rollout_data.select_idxs(replay_idxs)
                    if len(replay_rollout_data) > 0:
                        for i in range(len(replay_rollout_data)):
                            self.replay_buffer.put(replay_rollout_data[i:i+1])
                
                finished_rollout_data.non_tensor_batch.pop("raw_prompt_ids")
                finished_rollout_data.non_tensor_batch.pop("raw_response_ids")
                if self.config.psrl.log_prob.enable_inference_engine_log_prob:
                    device = finished_rollout_data.batch["input_ids"].device
                    rollout_log_probs = finished_rollout_data.non_tensor_batch.pop("rollout_log_probs", None)
                    rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(device)
                    rollout_log_probs = rollout_log_probs.to(torch.float32)
                    finished_rollout_data.batch["rollout_log_probs"] = rollout_log_probs
                
                if self.rollout_n > 1:
                    sample_ids = finished_rollout_data.non_tensor_batch["parent_id"]
                else:
                    sample_ids = finished_rollout_data.non_tensor_batch["uid"]
                request_ids = finished_rollout_data.non_tensor_batch["uid"]
                rollout_instance_ids = finished_rollout_data.non_tensor_batch["rollout_instance_id"]
                for i, (sample_id, request_id, rollout_instance_id) in enumerate(zip(sample_ids, request_ids, rollout_instance_ids)):
                    assert sample_id in self.rollout_request_buffer, \
                        f"Sample ID {sample_id} not found in rollout request buffer," \
                        " maybe the number of finished child requests have met `alg_rollout_n`," \
                        " and the sample has been removed from the buffer."
                    
                    request_data = self.rollout_request_buffer[sample_id]
                    response_data = finished_rollout_data[i:i+1]
                    merge_request_data = response_data.union(request_data)
                    self.rollout_parent_counter[sample_id] = self.rollout_parent_counter.get(sample_id, 0) + 1
                    
                    # 2025.7.9 先实现 rollout 完成就丢弃
                    # 如果在 rollout 完成就丢弃，则在此时通知 rollout server 丢弃 sample_id 对应的剩余子请求
                    # 可能不一定要在这删除 sample_id，允许更多子请求完成 gen，在 reward 计算完再统计 + 删除
                    if (
                        self.config.psrl.rollout_test.redundant_rollout.enable and
                        self.config.psrl.rollout_test.redundant_rollout.abort_stage == "rollout" and
                        self.rollout_parent_counter[sample_id] == self.alg_rollout_n
                    ):
                        # Required children of this sample_id have been processed, remove it from the buffer
                        ray.get(self.rollout_server_ref.exec_command(RolloutCommand(
                            CommandType.ABORT,
                            parent_ids=sample_id,
                        )).remote())
                        del self.rollout_request_buffer[sample_id]
                        del self.rollout_parent_counter[sample_id]
                    
                    # TODO: change to select instead of pop
                    batch_keys_to_pop = ["prompts", "attention_mask", "responses"]
                    non_tensor_batch_keys_to_pop = ["reward_model"]
                    if "extra_info" in merge_request_data.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("extra_info")
                    if "data_source" in merge_request_data.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("data_source")
                    reward_input = merge_request_data.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )
                    
                    # TODO: 改写 reward 计算逻辑
                    if self.use_rm:
                        pass
                    elif self.config.reward_model.launch_reward_fn_async:
                        with log_dual_events("Launch async reward model score", psrl_logger, event_type=EventType.OTHER):
                            future_reward = compute_reward_async.remote(reward_input, self.config, self.tokenizer)
                            merge_request_data.union(reward_input)
                            if self.config.psrl.log_prob.enable_inference_engine_log_prob:
                                future_reward_buffer.append((merge_request_data, future_reward))
                            else:
                                merge_request_data.non_tensor_batch["future_reward"] = np.array([future_reward])
                    else:
                        with log_dual_events("Compute reward model score", psrl_logger, event_type=EventType.OTHER):
                            reward_tensor, reward_extra_infos_dict = compute_reward(reward_input, self.reward_fn)
                            merge_request_data.union(reward_input)
                            merge_request_data.batch["reward"] = reward_tensor
                            merge_request_data.meta_info["reward_extra_info_keys"] = list(reward_extra_infos_dict.keys())
                            merge_request_data.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        
                    if (
                        self.config.psrl.log_prob.enable_inference_engine_log_prob and
                        self.config.reward_model.launch_reward_fn_async
                    ):
                        ready_reward, _ = ray.wait([future_reward for _, future_reward in future_reward_buffer])
                        unfinished_reward_buffer = []
                        futures = []
                        for request_data, future_reward in future_reward_buffer:
                            if future_reward in ready_reward:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                                request_data.batch["reward"] = reward_tensor
                                request_data.meta_info["reward_extra_info_keys"] = list(reward_extra_infos_dict.keys())
                                request_data.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                                # Notify the PS worker to store the request data and add count for its parent request
                                # If all child requests are completed, occupy the request
                                futures.append(self.ps_handle.store_and_maybe_occupy_rollout_instance_request.remote(
                                    rollout_instance_id=int(rollout_instance_id),
                                    request_id=str(request_id),
                                    data=request_data,
                                    parent_id=sample_id if self.rollout_n > 1 else None,
                                ))
                            else:
                                unfinished_reward_buffer.append((request_data, future_reward))
                        future_reward_buffer = unfinished_reward_buffer
                        with log_dual_events("Occupy requests", psrl_logger, event_type=EventType.OTHER):
                            ray.get(futures)
                    else:
                        with log_dual_events("Occupy requests", psrl_logger, event_type=EventType.OTHER):
                            futures = []
                            # Occupy the request in the PS worker
                            futures.append(self.ps_handle.store_and_maybe_occupy_rollout_instance_request.remote(
                                rollout_instance_id=int(rollout_instance_id),
                                request_id=str(request_id),
                                data=merge_request_data,
                                parent_id=sample_id if self.rollout_n > 1 else None,
                            ))
                            ray.get(futures)
                    
            except ray.util.queue.Empty:
                if self._reward_shutdown:
                    psrl_logger.info("Reward computing thread shutdown, exiting.")
                    break
        
        if future_reward_buffer:
            ready_reward = ray.get([future_reward for _, future_reward in future_reward_buffer])
            for request_data, future_reward in future_reward_buffer:
                if future_reward in ready_reward:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    request_data.batch["reward"] = reward_tensor
                    request_data.meta_info["reward_extra_info_keys"] = list(reward_extra_infos_dict.keys())
                    request_data.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                    if self.config.gen_actor_rollout_ref.rollout.n > 1:
                        sample_id = request_data.non_tensor_batch["parent_id"][0]
                    else:
                        sample_id = request_data.non_tensor_batch["uid"][0]
                    request_id = request_data.non_tensor_batch["uid"][0]
                    with log_dual_events("Occupy requests", psrl_logger, event_type=EventType.OTHER):
                        futures = []
                        # Occupy the request in the PS worker
                        futures.append(self.ps_handle.occupy_rollout_instance_request.remote(
                            rollout_instance_id=int(rollout_instance_id),
                            request_id=str(request_id),
                            data=request_data,
                            parent_id=sample_id if self.rollout_n > 1 else None,
                        ))
                        ray.get(futures)

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
        return self.total_training_steps