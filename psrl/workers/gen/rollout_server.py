from ast import Continue
import os
import logging
import time
import enum
import heapq
import threading
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

import ray
import numpy as np
from ray.util.queue import Queue as RayQueue

from verl import DataProto

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

class CommandType(Enum):
    STOP = enum.auto()
    SYNC = enum.auto()
    SHUTDOWN = enum.auto()
    ABORT = enum.auto()
    RESUME = enum.auto()
    CHECK = enum.auto()

@dataclass
class RolloutCommand:
    command_type: CommandType
    args: dict
    meta_data: dict
    
    def __init__(self, command_type, **kwargs):
        self.command_type = command_type
        for key, value in kwargs.items():
            if key == "meta_data":
                if not isinstance(value, dict):
                    raise ValueError(f"meta_data must be a dict, got {type(value)}")
                self.meta_data = value
            else:
                self.args[key] = value

    def __getattr__(self, item):
        if item in self.args:
            return self.args[item]
        elif item in self.meta_data:
            return self.meta_data[item]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
    
    def __setattr__(self, key, value):
        if key in self.args:
            self.args[key] = value
        else:
            self.meta_data[key] = value    

@dataclass
class RequestTracker:
    _instance_to_running_requests: dict[int, list[tuple[int, int]]] = defaultdict(list)  # Mapping from instance_id to request_ids
    _instance_to_updated_version: dict[int, int] = {}
    
    _running_request_id_to_metainfo: dict[int, tuple[int, int]] = {}  # Mapping from request_id to (instance_id, version_tag)
    _version_to_running_request_ids: dict[int, set[int]] = defaultdict(set)  # Mapping from version_tag to request_ids
    _finished_request_id_to_metainfo: dict[int, tuple[int, int]] = {}  # Mapping from finished request_id to (instance_id, version_tag)
    _version_to_finished_request_ids: dict[int, set[int]] = defaultdict(set)  # Mapping from version_tag to finished request_ids
    _interrupted_request_id_to_metainfo: dict[int, tuple[int, int]] = {}  # Mapping from interrupted request_id to (instance_id, version_tag)
    _version_to_interrupted_request_ids: dict[int, set[int]] = defaultdict(set)  # Mapping from version_tag to finished request_ids
    
    def cache_instance_version(self, instance_id, version_tag):
        self._instance_to_updated_version[instance_id] = version_tag
    
    def get_instance_version(self, instance_id):
        return self._instance_to_updated_version[instance_id]
    
    def add_request(self, request_id, instance_id, version_tag):
        """Add a request to the tracking structures."""
        if request_id in self._running_request_id_to_metainfo:
            raise ValueError(f"Request ID {request_id} already exists in request tracker.")
        self._version_to_running_request_ids[version_tag].add(request_id)
        heapq.heappush(self._instance_to_running_requests[instance_id], (version_tag, request_id))
        self._running_request_id_to_metainfo[request_id] = (instance_id, version_tag)
    
    def abort_requests(self, request_ids):
        """Abort requests from the tracking structures."""
        if isinstance(request_ids, int):
            request_ids = [request_ids]
        
        version_tags = [self.get_version_tag(request_id) for request_id in request_ids]
        instance_ids = [self.get_instance_id(request_id) for request_id in request_ids]

        for request_id, version_tag, instance_id in zip(request_ids, version_tags, instance_ids):
            if request_id in self._running_request_id_to_metainfo:
                self._running_request_id_to_metainfo.pop(request_id, None)
            if version_tag in self._version_to_running_request_ids:
                self._version_to_running_request_ids[version_tag].discard(request_id)
            if instance_id in self._instance_to_running_requests:
                self._instance_to_running_requests[instance_id] = [
                    (v_tag, req_id) for v_tag, req_id in self._instance_to_running_requests[instance_id]
                    if req_id != request_id
                ]
    
    def finish_requests(self, request_ids):
        """Mark requests as finished in the tracking structures."""
        if isinstance(request_ids, int):
            request_ids = [request_ids]
        
        version_tags = [self.get_version_tag(request_id) for request_id in request_ids]
        instance_ids = [self.get_instance_id(request_id) for request_id in request_ids]

        for request_id, version_tag, instance_id in zip(request_ids, version_tags, instance_ids):
            if request_id in self._running_request_id_to_metainfo:
                self._running_request_id_to_metainfo.pop(request_id, None)
            if version_tag in self._version_to_running_request_ids:
                self._version_to_running_request_ids[version_tag].discard(request_id)
            if instance_id in self._instance_to_running_requests:
                self._instance_to_running_requests[instance_id] = [
                    (v_tag, req_id) for v_tag, req_id in self._instance_to_running_requests[instance_id]
                    if req_id != request_id
                ]
            # Add to finished requests
            self._version_to_finished_request_ids[version_tag].add(request_id)
            self._finished_request_id_to_metainfo[request_id] = (instance_id, version_tag)

    def interrupt_requests(self, request_ids, update_version_tag=False):
        """Mark requests as interrupted in the tracking structures."""
        if isinstance(request_ids, int):
            request_ids = [request_ids]
        
        version_tags = [self.get_version_tag(request_id) for request_id in request_ids]
        instance_ids = [self.get_instance_id(request_id) for request_id in request_ids]

        for request_id, version_tag, instance_id in zip(request_ids, version_tags, instance_ids):
            if request_id in self._running_request_id_to_metainfo:
                self._running_request_id_to_metainfo.pop(request_id, None)
            if version_tag in self._version_to_running_request_ids:
                self._version_to_running_request_ids[version_tag].discard(request_id)
            if instance_id in self._instance_to_running_requests:
                self._instance_to_running_requests[instance_id] = [
                    (v_tag, req_id) for v_tag, req_id in self._instance_to_running_requests[instance_id]
                    if req_id != request_id
                ]
            # Add to interrupted requests
            if update_version_tag:
                assert instance_id in self._instance_to_updated_version, \
                    f"Instance {instance_id} does not have cached version tag."
                version_tag = self._instance_to_updated_version[instance_id]
            self._version_to_interrupted_request_ids[version_tag].add(request_id)
            self._interrupted_request_id_to_metainfo[request_id] = (instance_id, version_tag)
    
    def get_all_child_requets(self, parent_id, rollout_n):
        """Get all child requests for a given request_id."""
        child_requests = []
        for child_id in range(rollout_n):
            child_request_id = parent_id * rollout_n + child_id
            if child_request_id in self._running_request_id_to_metainfo:
                child_requests.append(child_request_id)
            if child_request_id in self._finished_request_id_to_metainfo:
                child_requests.append(child_request_id)
            if child_request_id in self._interrupted_request_id_to_metainfo:
                child_requests.append(child_request_id)
        return child_requests
    
    def has_request(self, request_id) -> bool:
        """Check if a request_id exists in the tracker."""
        return request_id in self._running_request_id_to_metainfo

    def get_request_metainfo(self, request_id):
        """Get the metainfo (instance_id, version_tag) for a given request_id."""
        if request_id in self._running_request_id_to_metainfo:
            return self._running_request_id_to_metainfo[request_id]
        else:
            raise ValueError(f"Request ID {request_id} not found in request tracker.")
    
    def is_aborted(self, request_ids):
        """Check if the request_ids are aborted."""
        if isinstance(request_ids, int):
            return request_ids not in self._running_request_id_to_metainfo
        
        if isinstance(request_ids, list):
            return [self.is_aborted(request_id) for request_id in request_ids]

    def get_version_tag(self, request_id):
        """Get the version tag for a given request_id."""
        if request_id in self._running_request_id_to_metainfo:
            return self._running_request_id_to_metainfo[request_id][1]
        else:
            raise ValueError(f"Request ID {request_id} not found in request tracker.")
    
    def get_instance_id(self, request_id):
        """Get the instance ID for a given request_id."""
        if request_id in self._running_request_id_to_metainfo:
            return self._running_request_id_to_metainfo[request_id][0]
        else:
            raise ValueError(f"Request ID {request_id} not found in request tracker.")
    
    def get_running_requests_of_version(self, version_tag):
        """Get all request IDs for a given version tag."""
        if version_tag in self._version_to_running_request_ids:
            return list(self._version_to_running_request_ids[version_tag])
        else:
            raise ValueError(f"Version tag {version_tag} not found in request tracker.")
    
    def get_all_requests_of_version(self, version_tag):
        """Get all request IDs for a given version tag, including finished and interrupted requests."""
        return list(self._version_to_running_request_ids.get(version_tag, [])) + \
                list(self._version_to_finished_request_ids.get(version_tag, [])) + \
                list(self._version_to_interrupted_request_ids.get(version_tag, []))
    
    def get_running_requests_of_instance(self, instance_id):
        """Get all request IDs for a given instance ID."""
        if instance_id in self._instance_to_running_requests:
            return [req_id for _, req_id in self._instance_to_running_requests[instance_id]]
        else:
            raise ValueError(f"Instance ID {instance_id} not found in request tracker.")
    
    def get_min_request_version_of_instance(self, instance_id):
        """Get the minimum version tag of requests for a given instance ID."""
        if instance_id in self._instance_to_running_requests and self._instance_to_running_requests[instance_id]:
            return self._instance_to_running_requests[instance_id][0][0]

@ray.remote
class RolloutServer:
    def __init__(
        self,
        config,
        rollout_wg_list,
        rollout_scheduler_cls,
        data_queue,
        rollout_queue,
        replay_buffer,
    ):
        self.config = config
        self.rank_0_is_model_owner = (
            self.config.gen_actor_rollout_ref.rollout.tensor_model_parallel_size *
                self.config.gen_actor_rollout_ref.rollout.pipeline_model_parallel_size > 1 and
            self.config.gen_actor_rollout_ref.rollout.mode == "psrl_async"
        )
        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_num = len(rollout_wg_list)
        self.rollout_scheduler = rollout_scheduler_cls(self.rollout_wg_num)
        self.staleness = self.config.psrl.staleness
        
        if self.config.psrl.rollout_test.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_test.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_test.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        
        self.data_queue = data_queue
        self.rollout_queue = rollout_queue
        self.replay_buffer = replay_buffer
        self.command_queue = RayQueue(maxsize=8)  # For async commands like shutdown

        self.interrupted_request_num = 0

        self._running = False
        self._paused = False
        self._interrupt_data_queue = False

        self._threads = []

        self._command_results = {}
        self._command_counter = 0
        
        self._request_tracker = RequestTracker() # Request management
        self._request_counter = 0
        
        self._abort_request_ids = []
    
    def abort_requests(self, request_ids):
        """Abort requests from the tracker."""
        self._request_tracker.abort_requests(request_ids)
    
    def finish_requests(self, request_ids):
        """Mark requests as finished in the tracker."""
        self._request_tracker.finish_requests(request_ids)
    
    def interrupt_requests(self, request_ids):
        """Mark requests as interrupted in the tracker."""
        self._request_tracker.interrupt_requests(request_ids)
    
    def is_aborted(self, request_ids):
        """Check if the request_ids are aborted."""
        return self._request_tracker.is_aborted(request_ids)
    
    def start_server(self):
        if self._running:
            return
        
        self._running = True
        futures = []
        for i in range(self.config.psrl.deployment.n_rollout_instances):
            if self.rank_0_is_model_owner:
                futures.append(self.rollout_wg_list[i].execute_rank_zero_async("busy_loop_generate_sequences", rollout_queue=self.rollout_queue))
            else:
                futures.extend(self.rollout_wg_list[i].execute_all_async("busy_loop_generate_sequences", rollout_queue=self.rollout_queue))
        ray.get(futures)

        self._data_queue_lock = threading.Lock()
        self._interrupt_data_queue = False

        event_thread = threading.Thread(
            target=self._background_event_handler,
            name="rollout_event_thread",
            daemon=True,
        )

        event_thread.start()
        self._threads = [event_thread]
    
    def shutdown_server(self):
        if not self._running:
            return
        
        psrl_logger.info("Shutdown rollout server...")
        self._running = False

        for thread in self._threads:
            if thread.is_alive():
                thread.join()
        self._threads = []

        psrl_logger.info("Rollout server shutdown.")

    def stop_server(self):
        psrl_logger.info(f"Waiting for {self.rollout_wg_num} workers to stop...")
        self.exec_command(RolloutCommand(CommandType.STOP))
        psrl_logger.info(f"Rollout server stopped.")
        return self.interrupted_request_num

    def resume_server(self):
        if self._paused:
            psrl_logger.info("Resuming rollout server...")
            self.exec_command(RolloutCommand(CommandType.RESUME))
            self._paused = False

    # similar to exec_command_async
    def add_command(self, command):
        self.command_queue.put(command)

    def exec_command(self, command, timeout=None):
        command_id = self._command_counter
        self._command_counter += 1

        command.meta_data["id"] = command_id
        self._command_results[command_id] = {"completed": False, "result": None}
        self.command_queue.put(command)

        start_time = time.time()
        while not self._command_results[command_id]["completed"]:
            if timeout is not None and time.time() - start_time > timeout:
                self._command_results.pop(command_id, None)
                return None
            time.sleep(0.01)

        psrl_logger.debug(f"Command {command_id} completed with result: {self._command_results[command_id]['result']}")
        result = self._command_results.pop(command_id, None)
        self._command_results.pop(command_id, None)
        
        return result
    
    def _complete_command(self, command_id, result):
        if command_id in self._command_results:
            self._command_results[command_id]["completed"] = True
            self._command_results[command_id]["result"] = result
            psrl_logger.debug(f"Command ID {command_id} completed with result: {result}")
        else:
            raise ValueError(f"Command ID {command_id} not found in results.")

    def schedule_requests(self, data):
        # 调度时携带version_tag
        schedule_plan = self.rollout_scheduler.schedule(data)
        for worker_id, requests in schedule_plan.items():
            if requests is None:
                continue
            version_tags = requests.non_tensor_batch["version_tag"]
            for i, version_tag in enumerate(version_tags):
                request_id = requests.non_tensor_batch["uid"][i]
                self._request_tracker.add_request(request_id, worker_id, version_tag)
                # heapq.heappush(self._instance_to_requests[worker_id], (version_tag, request_id))
                # self._request_id_to_metainfo[request_id] = (worker_id, version_tag)

            if self.rank_0_is_model_owner:
                self.rollout_wg_list[worker_id].execute_rank_zero_async("add_request", requests)
            else:
                self.rollout_wg_list[worker_id].execute_all_async("add_request", requests)

    def set_version_tag(self, request):
        # NOTE: naive implementation
        if self.config.psrl.rollout_test.redundant_rollout.enable:
            buffer_size = self.config.psrl.rollout_test.redundant_rollout.redundant_global_batch_size * self.rollout_n
        else:
            buffer_size = self.config.psrl.staleness_buffer_entries * self.rollout_n
        if self._request_counter <= self.staleness * buffer_size:
            version_tag = 0
        else:
            version_tag = (self._request_counter - self.staleness * buffer_size) // buffer_size
        return version_tag

    def check_interrupt(self, instance_id, curr_ps_model_version, staleness):
        # TODO: this min version may not be the same as the one in running queue
        # min_version_tag = self._instance_to_requests[instance_id][0][0]
        min_version_tag = self._request_tracker.get_min_request_version_of_instance(instance_id)
        if curr_ps_model_version - min_version_tag <= staleness:
            return True

        instance_request_ids = self._request_tracker.get_running_requests_of_instance(instance_id)
        abort_version_to_request_nums: dict[int, int] = {}
        abort_parent_to_request_nums: dict[int, int] = {}
        self._abort_request_ids = []
        for request_id in instance_request_ids:
            parent_id = request_id // self.rollout_n
            version_tag = self._request_tracker.get_version_tag(request_id)
            if version_tag < curr_ps_model_version - staleness:
                abort_version_to_request_nums[version_tag] = abort_version_to_request_nums.get(version_tag, 0) + 1
                abort_parent_to_request_nums[parent_id] = abort_parent_to_request_nums.get(parent_id, 0) + 1
                self._abort_request_ids.append(request_id)

        for parent_id, request_num in abort_parent_to_request_nums.items():
            if len(self._request_tracker.get_all_child_requets(parent_id, self.rollout_n)) - request_num < self.alg_rollout_n:
                return False

        for version_tag in sorted(abort_version_to_request_nums.keys()):
            expected_request_num = np.sum([len(self._request_tracker.get_all_requests_of_version(v)) for v in range(version_tag, version_tag + staleness)])
            abort_request_num = np.sum([abort_version_to_request_nums[v] for v in range(version_tag, version_tag + staleness)])
            if expected_request_num - abort_request_num < self.config.psrl.staleness_buffer_entries * self.alg_rollout_n:
                return False
        return True

    def _background_event_handler(self):
        while self._running:
            # Command processing
            if not self.command_queue.empty():
                command = self.command_queue.get()

                assert isinstance(command, RolloutCommand), f"Expected RolloutCommand, got {type(command)}"

                command_type = command.command_type
                command_id = command.meta_data.get("id", None)
                command_args = command.args
                
                result = None

                if command_type == CommandType.STOP:
                    psrl_logger.debug(f"begin to interrupt data queue processing")
                    futures = []
                    for i in range(self.config.psrl.deployment.n_rollout_instances):
                        if self.rank_0_is_model_owner:
                            futures.append(self.rollout_wg_list[i].execute_rank_zero_async("interrupt_all_requests"))
                        else:
                            futures.append(self.rollout_wg_list[i].execute_all_async("interrupt_all_requests"))
                    interrupted_request_nums = ray.get(futures)
                    psrl_logger.debug(f"RolloutServer: Received STOP command, interrupted {interrupted_request_nums} requests")
                    self.interrupted_request_num = np.sum(interrupted_request_nums)
                    self._paused = True
                    result = self.interrupted_request_num
                elif command_type == CommandType.SYNC:
                    instance_id = command_args.get("instance_id", None)
                    curr_ps_model_version = command_args.get("curr_ps_model_version", None)
                    if instance_id is None or curr_ps_model_version is None:
                        raise ValueError("SYNC command must contain 'instance_id' and 'curr_ps_model_version' in args.")
                    
                    if self.config.psrl.rollout_test.partial_rollout.interrupt_as_prompt:
                        self._request_tracker.cache_instance_version(instance_id, curr_ps_model_version)
                    
                    # stop -> pull_model -> resume
                    psrl_logger.debug(f"RolloutServer: Received SYNC command for instance {instance_id}, stopping generation...")
                    future = None
                    if self.rank_0_is_model_owner:
                        future = self.rollout_wg_list[instance_id].execute_rank_zero_async("interrupt_all_requests")
                    else:
                        future = self.rollout_wg_list[instance_id].execute_all_async("interrupt_all_requests")
                    interrupted_request_num = ray.get(future)
                    psrl_logger.debug(f"RolloutServer: Instance {instance_id} interrupted {interrupted_request_num} requests")
                    
                    # Pull model
                    psrl_logger.debug(f"RolloutServer: Pulling model for instance {instance_id}...")
                    if self.rank_0_is_model_owner:
                        future = self.rollout_wg_list[instance_id].execute_rank_zero_async("pull_model")
                    else:
                        future = self.rollout_wg_list[instance_id].execute_all_async("pull_model")
                    ray.get(future)
                    psrl_logger.debug(f"RolloutServer: Instance {instance_id} pulled model")
                    
                    # Resume generation
                    psrl_logger.debug(f"RolloutServer: Resuming generation for instance {instance_id}...")
                    if self.rank_0_is_model_owner:
                        self.rollout_wg_list[instance_id].execute_rank_zero_async("resume_generate")
                    else:
                        self.rollout_wg_list[instance_id].execute_all_async("resume_generate")
                    psrl_logger.debug(f"RolloutServer: Instance {instance_id} resumed generation")
                elif command_type == CommandType.SHUTDOWN:
                    self._running = False
                    for i in range(self.config.psrl.deployment.n_rollout_instances):
                        if self.rank_0_is_model_owner:
                            self.rollout_wg_list[i].execute_rank_zero_async("shutdown_generate")
                        else:
                            self.rollout_wg_list[i].execute_all_async("shutdown_generate")
                elif command_type == CommandType.ABORT:
                    assert "parent_ids" in command_args or "uids" in command_args, \
                        "Abort command must contain either 'parent_ids' or 'uids' in args."
                    parent_ids = command_args.get("parent_ids", None)
                    uids = command_args.get("uids", None)

                    if parent_ids is None and uids is None:
                        raise ValueError("Abort command must contain either 'parent_ids' or 'uids' in args.")
                    abort_map_from_instance_to_requests: dict[int, list[int]] = defaultdict(list)
                    # Get child requests from parent_ids
                    if parent_ids is not None:
                        parent_ids = set(parent_ids)  # Ensure uniqueness
                        for parent_id in parent_ids:
                            for child_id in range(self.rollout_n):
                                request_id = parent_id * self.rollout_n + child_id
                                # if request_id in self._request_id_to_metainfo:
                                if self._request_tracker.has_request(request_id):
                                    # instance_id, version_tag = self._request_id_to_metainfo[request_id]
                                    instance_id, version_tag = self._request_tracker.get_request_metainfo(request_id)
                                    abort_map_from_instance_to_requests[instance_id].append(request_id)
                                    
                                    # self.remove_tracked_request(request_id, version_tag, instance_id)
                                    self._request_tracker.abort_requests(request_id)
                    # Get requests from uids
                    if uids is not None:
                        uids = set(uids)
                        for uid in uids:
                            # if uid in self._request_id_to_metainfo:
                            if self._request_tracker.has_request(uid):
                                # instance_id, version_tag = self._request_id_to_metainfo[uid]
                                instance_id, version_tag = self._request_tracker.get_request_metainfo(uid)
                                abort_map_from_instance_to_requests[instance_id].append(uid)
                                
                                # self.remove_tracked_request(request_id, version_tag, instance_id)
                                self._request_tracker.abort_requests(uid)
                    
                    futures = []
                    for i in range(self.config.psrl.deployment.n_rollout_instances):
                        abort_requests = abort_map_from_instance_to_requests.get(i, [])
                        if not abort_requests:
                            continue

                        if self.rank_0_is_model_owner:
                            futures.append(self.rollout_wg_list[i].execute_rank_zero_async("interrupt_requests", abort_requests))
                        else:
                            futures.append(self.rollout_wg_list[i].execute_all_async("interrupt_requests", abort_requests))
                    interrupted_request_nums = ray.get(futures)
                    self.interrupted_request_num = np.sum(interrupted_request_nums)
                    psrl_logger.debug(f"RolloutServer: Received ABORT command, interrupted {self.interrupted_request_num} requests")
                elif command_type == CommandType.RESUME:
                    instance_ids = command_args.get("instance_ids", None)
                    if instance_ids is None:
                        instance_ids = range(self.config.psrl.deployment.n_rollout_instances)
                    
                    for instance_id in instance_ids:
                        if self.rank_0_is_model_owner:
                            self.rollout_wg_list[instance_id].execute_rank_zero_async("resume_generate")
                        else:
                            self.rollout_wg_list[instance_id].execute_all_async("resume_generate")
                elif command_type == CommandType.CHECK:
                    # Check if the current model version is sufficient for the given buffer_id
                    buffer_id = command_args.get("buffer_id", None)
                    curr_ps_model_version = command_args.get("curr_ps_model_version", None)
                    
                    if buffer_id is None or curr_ps_model_version is None:
                        raise ValueError("CHECK command must contain 'buffer_id' and 'curr_ps_model_version' in args.")
                    # 通知 rollout server 进行丢弃检查和中断检查
                    # 丢弃：需要发送满的 buffer id，则 version_tag 为 buffer_id - S 的正在 generate 的请求全部舍弃
                    # 中断：需要 track 每个 instance 内包含的请求数（running queue 的请求数？）；以及每个 instance 的 version_tag（对应内部 version_tag 最小的请求）
                    # 获取当前 model_store 的版本，衡量是否可以中断
                    abort_request_version = buffer_id - self.staleness
                    # abort_requests = self._version_to_request_ids.get(abort_request_version, set())
                    abort_requests = self._request_tracker.get_running_requests_of_version(abort_request_version)
                    if abort_requests:
                        self.exec_command(RolloutCommand(
                            command_type=CommandType.ABORT,
                            uids=list(abort_requests),
                        ))
                    if self.config.psrl.rollout_test.partial_rollout.enable:
                        instance_to_request_num = {}
                        futures = []
                        for instance_id in range(self.config.psrl.deployment.n_rollout_instances):
                            if self.rank_0_is_model_owner:
                                waiting_and_running_queue_size_ref = self.rollout_wg_list[instance_id].execute_rank_zero_async("waiting_and_running_queue_size")
                            else:
                                waiting_and_running_queue_size_ref = self.rollout_wg_list[instance_id].execute_all_async("waiting_and_running_queue_size")
                            instance_to_request_num[instance_id] = waiting_and_running_queue_size_ref
                            futures.append(waiting_and_running_queue_size_ref)
                        ray.get(futures)

                        for instance_id, request_num in instance_to_request_num.items():
                            if request_num > self.config.psrl.rollout_test.partial_rollout.threshould:
                                continue
                            if (
                                self.config.psrl.rollout_test.partial_rollout.interrupt_as_prompt or
                                self.check_interrupt(instance_id, curr_ps_model_version, self.staleness)
                            ):
                                psrl_logger.debug(f"RolloutServer: Instance {instance_id} can be interrupted, current model version: {curr_ps_model_version}")
                                self._request_tracker.abort_requests(self._abort_request_ids)
                                self._abort_request_ids = []

                                self.add_command(RolloutCommand(
                                    command_type=CommandType.SYNC,
                                    instance_id=instance_id,
                                    curr_ps_model_version=curr_ps_model_version,
                                ))
                else:
                    raise ValueError(f"Unknown command type: {command_type}")

                if command_id is not None:
                    self._complete_command(command_id, result)
            
            # Data processing
            # 先处理 replay buffer 中的请求
            if not self.replay_buffer.empty() and not self._paused:
                replay_data = self.replay_buffer.get_nowait()
                
                assert replay_data is not None, "Replay buffer data should not be None."
                non_tensor_batch_keys = replay_data.non_tensor_batch.keys()
                assert "version_tag" and "rollout_instance_id" in non_tensor_batch_keys, \
                    "Replay buffer data must contain 'version_tag' and 'rollout_instance_id' in non_tensor_batch."
                self.schedule_requests(replay_data)

            # 处理 data_queue 中的请求
            if not self.data_queue.empty() and not self._paused:
                data = self.data_queue.get_nowait()
                
                if data is None:
                    psrl_logger.info(f"RolloutServer: Received `None` data, skipping scheduling.")
                    for i in range(self.config.psrl.deployment.n_rollout_instances):
                        if self.rank_0_is_model_owner:
                            self.rollout_wg_list[i].execute_rank_zero_async("add_request", None)
                        else:
                            self.rollout_wg_list[i].execute_all_async("add_request", None)
                    self._running = False
                    continue

                # data_queue 中的请求已经被复制 n 份，如果是子请求，会包含 parent_id 字段。每个请求都有唯一的 uid 字段
                # 1. 记录父子关系
                # 2. 确定每个请求的 version_tag
                # 3. 维护 version_tag -> request_id 的映射关系
                # 4. 记录已完成请求的 version_tag -> request_id 映射关系
                batch_size = len(data)
                request_list = data.chunk(chunks=batch_size)
                for request in request_list:
                    version_tag = self.set_version_tag(request)
                    # self._version_to_request_ids[version_tag].add(request["uid"])
                    request.non_tensor_batch["version_tag"] = np.array(version_tag)
                data = DataProto.concat(request_list)

                self.schedule_requests(data)
