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
        Schedule single data item to worker using round-robin.
        Assumes data length is always 1.
        """

        if len(data) != 1:
            raise ValueError(f"RoundRobinRolloutScheduler expects data length of 1, got {len(data)}")
        
        # Create schedule plan with current worker getting the data
        schedule_plan = {}
        for i in range(self.rollout_worker_num):
            if i == self.current_worker_index:
                schedule_plan[i] = data
        
        # Move to next worker for next schedule call
        self.current_worker_index = (self.current_worker_index + 1) % self.rollout_worker_num
        
        return schedule_plan