import ray
import warnings
from copy import deepcopy

from verl.single_controller.ray.base import func_generator, RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.base.megatron.worker import DistGlobalInfo, DistRankInfo
from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup


# NOTE(lhy): NVMegatronRayWorkerGroup is not compatible with FusedWorker
# because during its `__init__`, it will call the `_execute_remote_single_worker` method,
# but not specifying which worker in the FusedWorker to call.
# This conflict is solved after uniformly using the RayWorkerGroup (after this PR: https://github.com/volcengine/verl/pull/2895),
# we refactor the original NVMegatronRayWorkerGroup here as a temporary solution for verl 0.4.1.x.
class RefactoredNVMegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    def __init__(self, resource_pool: RayResourcePool, ray_cls_with_init: RayClassWithInitArgs, **kwargs):
        """
        Initialize the RefactoredNVMegatronRayWorkerGroup.

        Args:
            resource_pool (RayResourcePool): The resource pool containing worker resources
            ray_cls_with_init (RayClassWithInitArgs): The Ray class with initialization arguments
            **kwargs: Additional keyword arguments to pass to the parent class
        """
        super().__init__(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, **kwargs)
        if not self.fused_worker_used:
            self.init_megatron_info()
        
    def spawn_fused(self, prefix_set):
        """Create a dictionary of worker groups for fused workers.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        warnings.warn("RefactoredNVMegatronRayWorkerGroup spawn_fused is a workaround for verl 0.4.1.x and 0.5.x.")
        wg_dict = dict()
        for key in prefix_set:
            new_wg = deepcopy(self)
            new_wg._bind_worker_method(self.ray_cls_with_init.cls.raw_cls_dict[key], func_generator)
            new_wg.sub_cls_name = key
            new_wg.init_megatron_info() # NOTE(lhy): newly added to make sure the megatron rank and global info are initialized for each worker properly (e.g., actor and critic if colocated).
            wg_dict[key] = new_wg
        return wg_dict
    
    def init_megatron_info(self):
        """
        Initialize the megatron rank and global info.
        """
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name="get_megatron_rank_info")
        self._megatron_global_info: DistGlobalInfo = ray.get(self.execute_rank_zero_async(method_name="get_megatron_global_info"))