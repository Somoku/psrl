import ray
import os
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass


def deep_update(orig: Dict[Any, Any], new: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Recursively merge new into orig, where if both are dict, merge them recursively.
    Otherwise, directly overwrite orig with new.
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
    """
    A specification for a PS worker.
    """
    node_ip: Optional[str] = None # ip of the node
    node_id: Optional[str] = None # ray id of the node
    attached_gpu_id: Optional[int] = None # if attached_gpu_id is not None, the actor will be binded to the attached GPU, otherwise it will be binded to the CPU only
    
    def __init__(self, node_ip: Optional[str] = None, node_id: Optional[str] = None, attached_gpu_id: Optional[int] = None):
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
            options: Dictionary of options to update
        """
        deep_update(self._options, options)

    def __call__(
        self, 
        target_node_id: str,
        attached_gpu_id: Optional[int] = None
    ) -> Any:
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

        
        
        