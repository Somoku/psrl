from abc import ABC, abstractmethod

from verl import DataProto


class RolloutSchedulerBase(ABC):
    """
    Abstract base class for a rollout scheduler.
    """

    @abstractmethod
    def schedule(self, data) -> dict[int, DataProto]:
        """
        Get a rollout from the environment and agent.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class BatchRolloutScheduler(RolloutSchedulerBase):
    """
    Rollout scheduler that collects rollouts in batches.
    """

    def __init__(
        self,
        rollout_worker_num,
    ):
        self.rollout_worker_num = rollout_worker_num

    def schedule(self, data):
        request_num = len(data)
        request_num_per_worker = -(-request_num // self.rollout_worker_num)
        
        schedule_plan = {}
        for i in range(self.rollout_worker_num):
            start_index = i * request_num_per_worker
            end_index = min((i + 1) * request_num_per_worker, request_num)
            schedule_plan[i] = data[start_index:end_index]
        return schedule_plan

class RoundRobinRolloutScheduler(RolloutSchedulerBase):
    """
    Rollout scheduler that uses round-robin to assign single data items to workers.
    """

    def __init__(
        self,
        rollout_worker_num,
    ):
        self.rollout_worker_num = rollout_worker_num
        self.current_worker_index = 0

    def schedule(self, data):
        """
        Schedule data to worker using round-robin.
        """

        request_num = len(data)
        request_list = data.chunk(request_num)
        schedule_plan = {}
        
        schedule_num = 0
        for i, request in enumerate(request_list):
            if "rollout_instance_id" in request.non_tensor_batch.keys():
                worker_idx = int(request.non_tensor_batch["rollout_instance_id"])
                schedule_plan[worker_idx].append(request)
                continue

            worker_idx = (self.current_worker_index + i) % self.rollout_worker_num
            if worker_idx not in schedule_plan:
                schedule_plan[worker_idx] = []
            schedule_plan[worker_idx].append(request)
            schedule_num += 1

        for worker_idx in schedule_plan.keys():
            schedule_plan[worker_idx] = DataProto.concat(schedule_plan[worker_idx])
        
        self.current_worker_index = (self.current_worker_index + schedule_num) % self.rollout_worker_num
        
        return schedule_plan