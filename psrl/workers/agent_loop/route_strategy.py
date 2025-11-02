import os
import logging
import numpy as np
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, List

from verl import DataProto
from psrl.workers.gen.stats_collector import EngineStats

_ROUTE_STRATEGY_REGISTRY: Dict[str, Type['RouteStrategyBase']] = {}

def register_route_strategy(name: str):
    """Register a route strategy class with the given name.
    
    Args:
        name (str): Name to register the strategy under.
        
    Returns:
        function: Decorator function for registering the class.
    """
    def decorator(cls: Type['RouteStrategyBase']):
        if name in _ROUTE_STRATEGY_REGISTRY:
            raise ValueError(f"Route strategy '{name}' is already registered")
        _ROUTE_STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator

def get_route_strategy_class(name: str) -> Type['RouteStrategyBase']:
    """Get the route strategy class by name.
    
    Args:
        name (str): Name of the route strategy.
        
    Returns:
        Type[RouteStrategyBase]: The route strategy class.
        
    Raises:
        ValueError: If the strategy name is not registered.
    """
    if name not in _ROUTE_STRATEGY_REGISTRY:
        raise ValueError(f"Route strategy '{name}' is not registered. Available strategies: {list(_ROUTE_STRATEGY_REGISTRY.keys())}")
    return _ROUTE_STRATEGY_REGISTRY[name]

def list_available_route_strategies() -> list:
    """Get a list of all available route strategy names.
    
    Returns:
        list: List of registered strategy names.
    """
    return list(_ROUTE_STRATEGY_REGISTRY.keys())

class RouteStrategyBase(ABC):
    def __init__(self, n_instances: int, strategy_kwargs: dict = None):
        """Initialize with the number of worker instances.
        
        Args:
            n_instances (int): Total number of worker instances.
        """
        self.n_instances = n_instances
        self.strategy_kwargs = strategy_kwargs
        self.instance_to_engine_status = {i: EngineStats(
            instance_id=i,
            model_version=0,
            snapshot=EngineStats.get_default_snapshot(),
        ) for i in range(n_instances)}
        self.instance_to_time_record = {i: datetime.now() for i in range(n_instances)} # Track the last clock of each instance
        self.logger = strategy_kwargs.get("logger", logging.getLogger(__file__))
        self.logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))
    
    @abstractmethod
    def route(self, request: DataProto, candidates: Optional[List[int]]) -> Optional[int]:
        """Route a request to a specific worker instance.
        
        Args:
            request (DataProto): The request to route.
            candidates (Optional[List[int]]): List of candidate worker indices if any.
            
        Returns:
            int: Index of the selected worker instance.
        """
        pass
    
    def update_instance_to_engine_status(self, instance_to_engine_status: dict[int, EngineStats]):
        """Update the engine status with latest information from coordinator.
        
        Args:
            instance_to_engine_status (dict[int, EngineStats]): Latest engine status information.
        """
        for i in range(self.n_instances):
            if i in instance_to_engine_status:
                self.instance_to_engine_status[i] = instance_to_engine_status[i]
                
    def push_request(self, request: DataProto, instance_id: int):
        """Push a request to a specific worker instance.
        
        Args:
            request (DataProto): The request to push.
            instance_id (int): The index of the worker instance to push the request to.
        """
        self.instance_to_time_record[instance_id] = datetime.now()
    
    def pop_request(self, request: DataProto, instance_id: int):
        """Pop a request from a specific worker instance.
        
        Args:
            request (DataProto): The request to pop.
            instance_id (int): The index of the worker instance to pop the request from.
        """
        self.instance_to_time_record[instance_id] = datetime.now()
    
    def is_staled(self, instance_id: int, engine_status: EngineStats) -> bool:
        """Check if the engine status is stale.
        
        Args:
            instance_id (int): The index of the worker instance.
            engine_status (EngineStats): The engine status.
        """
        snapshot_time = datetime.fromisoformat(engine_status.snapshot["timestamp"])
        return self.instance_to_time_record[instance_id] - snapshot_time > timedelta(milliseconds=self.strategy_kwargs.get("snapshot_staleness_threshold_in_ms", 100))

@register_route_strategy("random")
class RandomRouteStrategy(RouteStrategyBase):
    """Randomly selects a worker instance for each request."""

    def __init__(self, n_instances: int, strategy_kwargs: dict = None):
        super().__init__(n_instances, strategy_kwargs)
        self.logger.info("Initialized RandomRouteStrategy")

    def route(self, request: DataProto, candidates: Optional[List[int]] = None) -> Optional[int]:
        if candidates is None:
            candidates = list(range(self.n_instances))
        if len(candidates) == 0:
            return None
        return np.random.choice(candidates)

@register_route_strategy("round_robin")
class RoundRobinRouteStrategy(RouteStrategyBase):
    """Routes requests in a round-robin fashion across worker instances."""

    def __init__(self, n_instances: int, strategy_kwargs: dict = None):
        super().__init__(n_instances, strategy_kwargs)
        self.curr_idx = 0
        self.logger.info("Initialized RoundRobinRouteStrategy")

    def route(self, request: DataProto, candidates: Optional[List[int]] = None) -> Optional[int]:
        if candidates is None:
            candidates = list(range(self.n_instances))
        if len(candidates) == 0:
            return None
        # choose from candidates in a round-robin manner
        idx = self.curr_idx
        idx = candidates[idx % len(candidates)]
        self.curr_idx = (self.curr_idx + 1) % self.n_instances
        return idx

@register_route_strategy("request_num_balance")
class RequestNumBalanceRouteStrategy(RouteStrategyBase):
    """Routes requests to the worker instance with the fewest pending requests."""
    
    def __init__(self, n_instances: int, strategy_kwargs: dict = None):
        super().__init__(n_instances, strategy_kwargs)
        self.instance_request_counts = {i: 0 for i in range(n_instances)}
        self.logger.info(f"Initialized RequestNumBalanceRouteStrategy with instance request counts {self.instance_request_counts}")

    def route(self, request: DataProto, candidates: Optional[List[int]] = None) -> Optional[int]:
        if candidates is None:
            candidates = list(range(self.n_instances))
        if len(candidates) == 0:
            return None
        idx = np.argmin([self.instance_request_counts[i] for i in candidates])
        self.logger.debug(f"Routing request {request.non_tensor_batch['uid']} among candidates {candidates} "
                          f"with workloads {[self.instance_request_counts[i] for i in candidates]}, "
                          f"selected instance {candidates[idx]} with workload {self.instance_request_counts[candidates[idx]]}")
        remaining_request_counts = [self.instance_request_counts[i] for i in candidates if i != candidates[idx]]
        if len(remaining_request_counts) > 0:
            remaining_max_request_count = max(remaining_request_counts)
            if self.instance_request_counts[candidates[idx]] > remaining_max_request_count:
                # Avoid overload, return None (currently not route to any instance)
                return None
        self.instance_request_counts[candidates[idx]] += 1
        return candidates[idx]
    
    def update_instance_to_engine_status(self, instance_to_engine_status: dict[int, EngineStats]):
        super().update_instance_to_engine_status(instance_to_engine_status)
        for i, engine_stats in instance_to_engine_status.items():
            self.instance_request_counts[i] = engine_stats.get_waiting_and_running_queue_size()

    def push_request(self, request: DataProto, instance_id: int):
        super().push_request(request, instance_id)
        # Do not update the request count here, it has been updated when the request is routed to the instance
    
    def pop_request(self, request: DataProto, instance_id: int):
        super().pop_request(request, instance_id)
        self.instance_request_counts[instance_id] -= 1
        
    def is_staled(self, instance_id: int, engine_status: EngineStats) -> bool:
        if super().is_staled(instance_id, engine_status):
            return True
        if engine_status.get_waiting_and_running_queue_size() != self.instance_request_counts[instance_id]:
            self.logger.debug(f"Instance {instance_id} collected engine status is stale, "
                              f"waiting and running queue size {engine_status.get_waiting_and_running_queue_size()} "
                              f"is not equal to the recorded request count {self.instance_request_counts[instance_id]}")
            return True
        return False

@register_route_strategy("throughput_balance")
class ThroughputBalanceRouteStrategy(RouteStrategyBase):
    """Routes requests to the worker instance with optimal throughput balance."""
    
    def __init__(self, n_instances: int, strategy_kwargs: dict = None):
        super().__init__(n_instances, strategy_kwargs)
        # TODO(lhy)
        raise NotImplementedError("ThroughputBalanceRouteStrategy is not implemented")
    
    def route(self, request: DataProto, candidates: Optional[List[int]] = None) -> int:
        if candidates is None:
            candidates = list(range(self.n_instances))
        if len(candidates) == 0:
            return None
        # TODO(lhy)
        raise NotImplementedError("ThroughputBalanceRouteStrategy is not implemented")

