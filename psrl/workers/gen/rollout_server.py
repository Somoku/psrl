import os
import logging
import time
import enum
import threading
from enum import Enum

import ray
import numpy as np
from ray.util.queue import Queue as RayQueue

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

class CommandType(Enum):
    STOP = enum.auto()
    SYNC = enum.auto()
    SHUTDOWN = enum.auto()
    ABORT = enum.auto()
    RESUME = enum.auto()

@ray.remote
class RolloutServer:
    def __init__(
        self,
        config,
        rollout_wg_list,
        rollout_scheduler_cls,
        data_queue,
        rollout_queue,
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
        
        self.data_queue = data_queue
        self.rollout_queue = rollout_queue
        self.command_queue = RayQueue(maxsize=8)  # For async commands like shutdown
        self.interrupted_request_num = 0

        self._running = False
        self._paused = False

        self._loop = None
        self._tasks = []

        self._command_results = {}
        self._command_counter = 0
    
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
        self.exec_command({"type": CommandType.STOP})
        psrl_logger.info(f"Rollout server stopped.")
        return self.interrupted_request_num

    def resume_server(self):
        if self._paused:
            psrl_logger.info("Resuming rollout server...")
            self.exec_command({"type": CommandType.RESUME})
            self._paused = False

    # similar to exec_command_async
    def add_command(self, command):
        self.command_queue.put(command)

    def exec_command(self, command, timeout=None):
        command_id = self._command_counter
        self._command_counter += 1

        command_with_id = command.copy()
        command_with_id["id"] = command_id

        self._command_results[command_id] = {"completed": False, "result": None}

        self.command_queue.put(command_with_id)

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
        schedule_plan = self.rollout_scheduler.schedule(data)
        for worker_id, requests in schedule_plan.items():
            if requests is None:
                continue
            if self.rank_0_is_model_owner:
                self.rollout_wg_list[worker_id].execute_rank_zero_async("add_request", requests)
            else:
                self.rollout_wg_list[worker_id].execute_all_async("add_request", requests)

    def _background_event_handler(self):
        while self._running:
            # Command processing
            if not self.command_queue.empty():
                command = self.command_queue.get()
                
                assert isinstance(command, dict), f"Expected command to be a dict, got {type(command)}"

                cmd_type = command.get("type")
                command_id = command.get("id")

                result = None

                if cmd_type == CommandType.STOP:
                    psrl_logger.debug(f"begin to interrupt data queue processing")
                    futures = []
                    for i in range(self.config.psrl.deployment.n_rollout_instances):
                        if self.rank_0_is_model_owner:
                            self.rollout_wg_list[i].execute_rank_zero_async("interrupt_all_requests")
                        else:
                            self.rollout_wg_list[i].execute_all_async("interrupt_all_requests")
                    interrupted_request_nums = ray.get(futures)
                    psrl_logger.debug(f"RolloutServer: Received STOP command, interrupted {interrupted_request_nums} requests")
                    self.interrupted_request_num = np.sum(interrupted_request_nums)
                    self._paused = True
                    result = self.interrupted_request_num
                elif cmd_type == CommandType.SHUTDOWN:
                    self._running = False
                    for i in range(self.config.psrl.deployment.n_rollout_instances):
                        if self.rank_0_is_model_owner:
                            self.rollout_wg_list[i].execute_rank_zero_async("shutdown_generate")
                        else:
                            self.rollout_wg_list[i].execute_all_async("shutdown_generate")
                elif cmd_type == CommandType.ABORT:
                    raise NotImplementedError
                elif cmd_type == CommandType.RESUME:
                    for i in range(self.config.psrl.deployment.n_rollout_instances):
                        if self.rank_0_is_model_owner:
                            self.rollout_wg_list[i].execute_rank_zero_async("resume_generate")
                        else:
                            self.rollout_wg_list[i].execute_all_async("resume_generate")
                else:
                    raise ValueError(f"Unknown command type: {cmd_type}")

                if command_id is not None:
                    self._complete_command(command_id, result)
            
            # Data processing
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

                self.schedule_requests(data)
