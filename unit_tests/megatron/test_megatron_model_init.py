"""
Megatron training client Actor example
Focuses on core Megatron model loading functionality
"""

import os
import ray
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from megatron.core import parallel_state as mpu

# Import verl utilities
from verl.utils.megatron_utils import get_model
from verl.utils.device import get_device_name
from verl.models.mcore import init_mcore_model
from verl.utils.torch_dtypes import PrecisionType

QWEN_MODEL_PATH = os.environ.get("PSRL_WORKSPACE", "/tmp") + "/models/Qwen2.5-0.5B-Instruct"

@ray.remote(num_cpus=1)
class MegatronClient:
    def __init__(self, rank, world_size, model_path=QWEN_MODEL_PATH):
        # Set environment variables
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        
        self.rank = rank
        self.world_size = world_size
        self.model_path = model_path
        
        print(f"[Rank {rank}] Initializing MegatronClient")
        
        # Initialize distributed training
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        
        # Initialize Megatron parallel state
        self._init_megatron_parallel()
        
        # Initialize model
        self._init_model()
        
        print(f"[Rank {rank}] Initialization completed")
    
    def _init_megatron_parallel(self):
        """Initialize Megatron parallel state"""
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,  # No tensor parallelism
            pipeline_model_parallel_size=1,  # No pipeline parallelism
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            nccl_communicator_config_path=None,
        )
        print(f"[Rank {self.rank}] Megatron parallel initialized")
    
    def _init_model(self):
        """Initialize Megatron model"""
        from transformers import AutoConfig
        from verl.models.mcore import hf_to_mcore_config
        from verl.utils.fs import copy_to_local
        
        # Copy model to local
        local_path = copy_to_local(self.model_path)
        
        # Get HuggingFace config
        hf_config = AutoConfig.from_pretrained(local_path, trust_remote_code=False)
        
        # Convert to Megatron config
        dtype = PrecisionType.to_dtype(torch.bfloat16)
        tf_config = hf_to_mcore_config(hf_config, dtype)
        
        print(f"[Rank {self.rank}] Config loaded: {hf_config.model_type}")
        
        def model_provider(pre_process, post_process):
            """Model provider function"""
            model = init_mcore_model(
                tf_config, 
                hf_config, 
                pre_process, 
                post_process, 
                share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
                value=False
            )
            model.to(get_device_name())
            return model
        
        # Initialize Megatron model
        self.model = get_model(
            model_provider,
            wrap_with_ddp=True,
            use_distributed_optimizer=False,
        )
        
        print(f"[Rank {self.rank}] Model initialized with {len(self.model)} modules")
    
    def get_model_info(self):
        """Get model information"""
        if hasattr(self, 'model') and self.model:
            total_params = sum(p.numel() for p in self.model[0].parameters())
            return {
                "rank": self.rank,
                "total_parameters": total_params,
                "model_type": "megatron",
                "world_size": self.world_size
            }
        return {"error": "Model not initialized"}
    
    def shutdown(self):
        """Shutdown client"""
        print(f"[Rank {self.rank}] Shutting down")
        if dist.is_initialized():
            dist.destroy_process_group()
        return True

def test_megatron():
    """Test Megatron client"""
    ray.init(ignore_reinit_error=True)
    
    num_workers = 2
    print(f"Testing MegatronClient with {num_workers} workers")
    
    # Create clients
    clients = [
        MegatronClient.options(num_gpus=1).remote(
            rank=rank, 
            world_size=num_workers
        )
        for rank in range(num_workers)
    ]
    
    # Get model information
    model_infos = ray.get([client.get_model_info.remote() for client in clients])
    for info in model_infos:
        print(f"Model info: {info}")
    
    # Shutdown clients
    ray.get([client.shutdown.remote() for client in clients])
    ray.shutdown()
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_megatron()
