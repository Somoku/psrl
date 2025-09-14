import ray
import os
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass


def deep_update(orig: Dict[Any, Any], new: Dict[Any, Any]) -> Dict[Any, Any]:
    """Recursively merge two dictionaries.
    
    If both orig and new have the same key and both values are dictionaries,
    merge them recursively. Otherwise, overwrite orig with new.
    
    Args:
        orig (Dict[Any, Any]): The original dictionary to update
        new (Dict[Any, Any]): The new dictionary to merge in
        
    Returns:
        Dict[Any, Any]: The updated original dictionary
    """
    for key, val in new.items():
        if (
            key in orig
            and isinstance(orig[key], dict)
            and isinstance(val, dict)
        ):
            deep_update(orig[key], val)
        else:
            orig[key] = val
    return orig


@dataclass
class PSResourceSpec:
    """Specification for a PS worker resource allocation.
    
    This class defines where a PS worker should be deployed, including
    the target node and optional GPU binding.
    
    Attributes:
        node_ip (Optional[str]): IP address of the target node
        node_id (Optional[str]): Ray node ID of the target node  
        attached_gpu_id (Optional[int]): GPU ID to bind the worker to, None for CPU-only
    """
    node_ip: Optional[str] = None # ip of the node
    node_id: Optional[str] = None # ray id of the node
    attached_gpu_id: Optional[int] = None # if attached_gpu_id is not None, the actor will be binded to the attached GPU, otherwise it will be binded to the CPU only
    
    def __init__(self, node_ip: Optional[str] = None, node_id: Optional[str] = None, attached_gpu_id: Optional[int] = None):
        """Initialize PSResourceSpec with node and GPU information.
        
        Either node_ip or node_id must be provided. If only one is given,
        the other will be automatically resolved using Ray's node information.
        
        Args:
            node_ip (Optional[str]): IP address of the target node
            node_id (Optional[str]): Ray node ID of the target node
            attached_gpu_id (Optional[int]): GPU ID to bind the worker to
        """
        if node_ip is None and node_id is None:
            raise ValueError("node_ip or node_id must be set")
        
        self.node_ip = node_ip
        self.node_id = node_id
        self.attached_gpu_id = attached_gpu_id

        ip_to_node_id = {node['NodeManagerAddress']: node['NodeID'] for node in ray.nodes()}
        node_id_to_ip = {node['NodeID']: node['NodeManagerAddress'] for node in ray.nodes()}
        
        if node_ip is None and node_id is None:
            raise ValueError("node_ip or node_id must be set")
        if node_ip is not None and node_id is None:
            assert node_ip in ip_to_node_id, f"node_ip {node_ip} not found in ray nodes"
            self.node_id = ip_to_node_id[node_ip]
        if node_id is not None and node_ip is None:
            assert node_id in node_id_to_ip, f"node_id {node_id} not found in ray nodes"
            self.node_ip = node_id_to_ip[node_id]
        if node_id is not None and node_ip is not None:
            assert node_id in node_id_to_ip, f"node_id {node_id} not found in ray nodes"
            assert node_id_to_ip[node_id] == node_ip, f"node_id {node_id} and node_ip {node_ip} mismatch"


@dataclass
class PSResourcePool:
    """A pool of PS resource specifications.
    
    This class represents a collection of PSResourceSpec objects that define
    the resources required for each PS worker.
    """
    ps_spec_list: List[PSResourceSpec]


class PSClassWithInitArgs:
    """A wrapper class for PS actors with initialization arguments."""

    def __init__(self, cls, *args, **kwargs) -> None:
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self._options = {}

    def update_options(self, options: Dict):
        """Update the PS actor creation options.

        Args:
            options (Dict): Dictionary of options to update. Will be merged with existing options
        """
        deep_update(self._options, options)

    def __call__(
        self, 
        target_node_id: str,
        attached_gpu_id: Optional[int] = None
    ) -> Any:
        """Create and deploy a PS actor on the specified node.
        
        Args:
            target_node_id (str): Ray node ID where the actor should be deployed
            attached_gpu_id (Optional[int]): GPU ID to bind the actor to
            
        Returns:
            Any: Ray actor handle for the created PS worker
        """
        num_gpus = 0.5 if attached_gpu_id is not None else 0
        options = {
            "num_cpus": 0, 
            "num_gpus": num_gpus, 
            "scheduling_strategy": NodeAffinitySchedulingStrategy(
                node_id=target_node_id, 
                soft=False
            )
        }
        if attached_gpu_id is not None:
            options["runtime_env"] = {
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "CUDA_VISIBLE_DEVICES": str(attached_gpu_id)
                }
            }
        deep_update(options, self._options)
        return self.cls.options(**options).remote(*self.args, **self.kwargs)
        

class PSWorkerGroup:
    """Group of PS workers for distributed parameter server operations.
    
    This class manages a collection of PS workers deployed across multiple nodes,
    providing methods to execute operations on all workers simultaneously.
    """
    def __init__(
        self, 
        resource_pool: PSResourcePool,
        ps_cls_with_init: PSClassWithInitArgs,
        **kwargs
    ) -> None:
        self._workers = []
        self._init_with_resource_pool(resource_pool=resource_pool, ps_cls_with_init=ps_cls_with_init)
        
    @property
    def world_size(self):
        """Number of workers in the group."""
        return len(self._workers)
    
    def distinguish_worker_by_method(self, callable_method: Callable):
        """Distinguish workers by the callable method.
        
        Args:
            callable_method: A callable method that takes a worker as input and returns a boolean value.
        """
        candidates = [worker for worker in self._workers if callable_method(worker)]
        assert len(candidates) == 1, f"Expected 1 worker, but got {len(candidates)} workers that satisfy the condition"
        return candidates[0]
        
    def _init_with_resource_pool(self, resource_pool: PSResourcePool, ps_cls_with_init: PSClassWithInitArgs) -> None:
        rank = -1
        for ps_spec in resource_pool.ps_spec_list:
            rank += 1
            env_vars = {
                "WORLD_SIZE": str(len(resource_pool.ps_spec_list)),
                "RANK": str(rank),
                "PS_NODE_IP": ps_spec.node_ip,
            }
            ps_cls_with_init.update_options({"runtime_env": {"env_vars": env_vars}, "name": f"PSWorker_{rank}"})
            # create a worker
            worker = ps_cls_with_init(target_node_id=ps_spec.node_id, attached_gpu_id=ps_spec.attached_gpu_id)
            self._workers.append(worker)
            
    def _execute_remote_single_worker(self, worker, method_name: str, *args, **kwargs):
        return getattr(worker, method_name).remote(*args, **kwargs)
            
    def execute_all_async(self, method_name: str, *args, **kwargs):
        """Execute a method on all workers asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        # Here, we assume that if all arguments in args and kwargs are lists,
        # and their lengths match len(self._workers), we'll distribute each
        # element in these lists to the corresponding worker
        # print(f"execute_all_async: method {method_name}({args}, {kwargs})")
        length = len(self._workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                # print(f"splitting args and kwargs into {length} shards")
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    result.append(self._execute_remote_single_worker(self._workers[i], method_name, *sliced_args, **sliced_kwargs))
                return result

        return [self._execute_remote_single_worker(worker, method_name, *args, **kwargs) for worker in self._workers]

        
        
        