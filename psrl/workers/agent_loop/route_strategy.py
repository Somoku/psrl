import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, List

from verl import DataProto

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
    @abstractmethod
    def __init__(self, n_instances: int):
        """Initialize with the number of worker instances.
        
        Args:
            n_instances (int): Total number of worker instances.
        """
        pass
    
    @abstractmethod
    def route(self, request: DataProto, candidates: Optional[List[int]]) -> int:
        """Route a request to a specific worker instance.
        
        Args:
            request (DataProto): The request to route.
            candidates (Optional[List[int]]): List of candidate worker indices if any.
            
        Returns:
            int: Index of the selected worker instance.
        """
        pass

@register_route_strategy("random")
class RandomRouteStrategy(RouteStrategyBase):
    """Randomly selects a worker instance for each request."""

    def __init__(self, n_instances: int):
        self.n_instances = n_instances

    def route(self, request: DataProto, candidates: Optional[List[int]] = None) -> int:
        if candidates is None:
            candidates = list(range(self.n_instances))
        return np.random.choice(candidates)

@register_route_strategy("round_robin")
class RoundRobinRouteStrategy(RouteStrategyBase):
    """Routes requests in a round-robin fashion across worker instances."""

    def __init__(self, n_instances: int):
        self.n_instances = n_instances
        self.curr_idx = 0

    def route(self, request: DataProto, candidates: Optional[List[int]] = None) -> int:
        if candidates is None:
            candidates = list(range(self.n_instances))
        # choose from candidates in a round-robin manner
        idx = self.curr_idx
        idx = candidates[idx % len(candidates)]
        self.curr_idx = (self.curr_idx + 1) % self.n_instances
        return idx

@register_route_strategy("request_num_balance")
class RequestNumBalanceRouteStrategy(RouteStrategyBase):
    """Routes requests to the worker instance with the fewest pending requests."""
    
    def __init__(self, n_instances: int):
        """Initialize with the number of worker instances.
        
        Args:
            n_instances (int): Total number of worker instances.
        """
        self.n_instances = n_instances
        self.instance_request_counts = {i: 0 for i in range(n_instances)}
    
    def update_instance_request_counts(self, counts: dict[int, int]):
        """Update the request counts for each worker instance.
        
        Args:
            counts (dict[int, int]): Mapping of instance ID to request count.
        """
        for idx, count in counts.items():
            assert 0 <= idx < self.n_instances, f"Instance index {idx} out of range [0, {self.n_instances})"
            self.instance_request_counts[idx] = count

    def route(self, request: DataProto, candidates: Optional[List[int]] = None) -> int:
        if candidates is None:
            candidates = list(range(self.n_instances))
        idx = np.argmin([self.instance_request_counts[i] for i in candidates])
        self.instance_request_counts[candidates[idx]] += 1
        return candidates[idx]
