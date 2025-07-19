from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Union

from verl import DataProto


class RolloutRouterBase(ABC):
    """
    Abstract base class for a rollout router.
    """

    @abstractmethod
    def route(
        self,
        data: DataProto,
        instance_running_status: Optional[dict[int, bool]] = None,
    ) -> dict[int, DataProto]:
        """
        Route the requests in data to different running instances based on the routing strategy.
        
        Args:
            data (DataProto): The data to be processed.
            instance_running_status (Optional[dict[int, bool]]): A map of instance IDs to their running status.

        Returns:
            dict[int, DataProto]: A mapping of instance IDs to the data assigned to them.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class BatchRolloutRouter(RolloutRouterBase):
    """
    Rollout router that route requests in batches to instances.
    
    This router divides the requests into batches and assigns each mini-batch to a instance.
    """

    def __init__(
        self,
        rollout_instance_num,
    ):
        """
        Initialize the BatchRolloutRouter with the number of rollout instances.
        
        Args:
            rollout_instance_num (int): The number of rollout instances to distribute requests to.
        """
        self.rollout_instance_num = rollout_instance_num

    def route(
        self,
        data: DataProto,
        instance_running_status: Optional[dict[int, bool]] = None,
    ):
        """Route requests in `data` to instances evenly."""
        if instance_running_status is not None:
            running_instances = [
                instance_id for instance_id, is_running in instance_running_status.items() if is_running]
        else:
            running_instances = list(range(self.rollout_instance_num))
        
        request_num = len(data)
        running_instance_num = len(running_instances)
        request_num_per_instance = -(-request_num // running_instance_num)  
        
        route_plan: dict[int, list[DataProto]] = defaultdict(list)
        for i, instance_id in enumerate(running_instances):
            start_index = i * request_num_per_instance
            end_index = min((i + 1) * request_num_per_instance, request_num)
            route_plan[instance_id] = data[start_index:end_index]
        return route_plan

class RoundRobinRolloutRouter(RolloutRouterBase):
    """
    Rollout router that uses round-robin to assign single data items to workers.
    """

    def __init__(
        self,
        rollout_instance_num,
    ):
        """
        Initialize the RoundRobinRolloutRouter with the number of rollout instances.
        
        Args:
            rollout_instance_num (int): The number of rollout instances to distribute requests to.
        """
        self.rollout_instance_num = rollout_instance_num
        self.current_instance_index = 0

    def route(
        self,
        data: DataProto,
        instance_running_status: Optional[dict[int, bool]] = None,
    ):
        """
        Schedule data to running instances using round-robin.
        """

        request_num = len(data)
        request_list = data.chunk(request_num)
        route_plan: dict[int, Union[list[DataProto], DataProto]] = defaultdict(list)
        
        for i, request in enumerate(request_list):
            if "rollout_instance_id" in request.non_tensor_batch.keys():
                instance_id = int(request.non_tensor_batch["rollout_instance_id"])
                if instance_running_status is not None:
                    # If instance_running_status is provided, check if the instance is running
                    assert instance_running_status[instance_id], \
                        f"Rollout instance {instance_id} is not in the running status map, but received request {request}. This should not happen."
                route_plan[instance_id].append(request)
                continue

            self.current_instance_index = (self.current_instance_index + 1) % self.rollout_instance_num
            if instance_running_status is not None:
                while not instance_running_status[self.current_instance_index]:
                    self.current_instance_index = (self.current_instance_index + 1) % self.rollout_instance_num

            route_plan[self.current_instance_index].append(request)

        for instance_id in route_plan.keys():
            route_plan[instance_id] = DataProto.concat(route_plan[instance_id])
        
        return route_plan