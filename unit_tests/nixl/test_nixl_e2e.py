import os
import time
import ray
import torch
import socket
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoConfig, AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy
from omegaconf import OmegaConf
from megatron.core import parallel_state as mpu

from verl.workers.megatron_workers import set_random_seed
from verl.utils.megatron_utils import get_model
from verl.utils.device import get_device_name
from verl.models.mcore import init_mcore_model, hf_to_mcore_config
from verl.utils.torch_dtypes import PrecisionType

from psrl.utils.nixl import NIXLClientType, NIXLInterface, NIXLMetaServer, NIXLStorageClient, GLOBAL_META_SERVER_NAME, GLOBAL_PORT_SCANNER
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.vllm_converter import convert_vllm_inplace
from psrl.utils.converter.fsdp_converter import convert_fsdp_inplace
from psrl.utils.converter.megatron_converter import convert_megatron_inplace
from psrl.workers.ps import PSWorkerGroup, PSClassWithInitArgs, PSResourcePool, PSResourceSpec, PSStorageWorker, PSStoragePlan

QWEN_MODEL_PATH = os.environ.get("PSRL_WORKSPACE") + "/models/Qwen2.5-0.5B-Instruct"
# QWEN_MODEL_PATH = os.environ.get("PSRL_WORKSPACE") + "/models/Qwen2.5-Math-7B"
# QWEN_MODEL_PATH = os.environ.get("PSRL_WORKSPACE") + "/models/Qwen2.5-3B-Instruct"

def make_dual_print(log_path, prefix=None):
    with open(log_path, "w") as f:
        pass
    def dual_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if prefix:
            msg = f"[{prefix}] {msg}"
        print(msg, **kwargs)
        with open(log_path, "a") as f:
            print(msg, file=f, **kwargs)
    return dual_print

@ray.remote
class MetaServerActor:
    def __init__(self, server_name, psrl_config, expected_agents, log_dir):
        self.server = NIXLMetaServer(server_name, psrl_config.nixl)
        self.expected_agents = expected_agents
        self.client_name = server_name
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.client_name}.log")
        self.print = make_dual_print(log_path, prefix=self.client_name)

    def init_finished(self):
        return True

    def protocol(self):
        self.print("step1: wait_for_client_shardings")  
        self.server.wait_for_client_shardings(self.expected_agents, timeout=120)
        # self.print(f"client_sharding_dicts: {self.server.client_sharding_dicts}")
        self.print("step2: make_unified_sharding")
        self.server.make_unified_sharding()
        # self.print(f"client_unified_sharding_dicts: {self.server.client_unified_sharding_dicts}")
        self.print("step3: notify_all_client_shardings")
        self.server.notify_all_client_shardings()
        self.print("step4: wait_for_client_infos")
        self.server.wait_for_client_infos(self.expected_agents, timeout=120)
        # self.print(f"client_infos: {self.server.client_infos}")
        self.print("step5: make_comm_plan")
        self.server.make_comm_plan()
        self.print("step6: notify_all_client_infos_and_comm_plan")
        self.server.notify_all_client_infos_and_comm_plan()
        self.print("step7: wait_for_client_temp_mappings")
        self.server.wait_for_client_temp_mappings(self.expected_agents, timeout=120)
        self.print("step8: notify_all_client_temp_mappings")
        self.server.notify_all_client_temp_mappings()
        self.print("protocol done.")
        return self.server.client_unified_sharding_dicts

    def shutdown(self):
        self.server.shutdown()

@ray.remote(num_cpus=1)
class TrainClientActor:
    def __init__(self, engine_type, rank, world_size, server_name, psrl_config, backend, torch_port, log_dir, nixl_interface: NIXLInterface, ps_for_push_worker_handles):
        assert engine_type in ["fsdp", "megatron"], f"engine {engine_type} is not supported"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(torch_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        self.engine_type = engine_type
        self.rank = rank
        self.world_size = world_size
        self.client_name = f"train_client_{rank}"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.client_name}.log")
        self.print = make_dual_print(log_path, prefix=self.client_name)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        self.print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        if engine_type == "fsdp":
            model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_PATH, torch_dtype=torch.float32)
            # Set all model parameters to 1
            for p in model.parameters():
                p.data.fill_(1)
            torch.manual_seed(42)
            self.model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=torch.cuda.current_device(),
                sync_module_states=False,
                device_mesh=init_device_mesh("cuda", mesh_shape=(world_size,))
            )
        elif engine_type == "megatron":
            self._init_megatron_parallel()
            set_random_seed(42)
            self._init_megatron_model()
        self.client = NIXLStorageClient(
            client_name=self.client_name,
            server_name=server_name,
            use_gpu=True,
            client_type=NIXLClientType.PUSH_SIDE,
            nixl_config=psrl_config.nixl,
            nixl_interface=nixl_interface
        )
        self.ps_for_push_worker_handles = ps_for_push_worker_handles
    
    def _init_megatron_parallel(self):
        """Initialize Megatron parallel state"""
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=2,  
            pipeline_model_parallel_size=2, 
            virtual_pipeline_model_parallel_size=2,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            nccl_communicator_config_path=None,
        )
        self.print(f"[Rank {self.rank}] Megatron parallel initialized")
    
    def _init_megatron_model(self):
        """Initialize Megatron model"""
        # Get HuggingFace config
        hf_config = AutoConfig.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=False)
        # Convert to Megatron config
        dtype = PrecisionType.to_dtype(torch.bfloat16)
        tf_config = hf_to_mcore_config(hf_config, dtype)
        self.print(f"[Rank {self.rank}] Config loaded: {hf_config.model_type}")
        
        def model_provider(pre_process, post_process, vp_stage=None):
            """Model provider function"""
            model = init_mcore_model(
                tf_config, 
                hf_config, 
                pre_process, 
                post_process, 
                share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
                value=False,
                vp_stage=vp_stage,
            )
            model.to(get_device_name())
            for p in model.parameters():
                p.data.fill_(1)
            return model
        
        # Initialize Megatron model
        self.model = get_model(
            model_provider,
            wrap_with_ddp=True,
            use_distributed_optimizer=True,
        )
        
        self.print(f"[Rank {self.rank}] Model initialized: {self.model}")
        # self.print(f"[Rank {self.rank}] Model state_dict keys: {[submodel.state_dict().keys() for submodel in self.model]}")
        # self.print(f"[Rank {self.rank}] Model named parameter keys: {[name for submodel in self.model for name, _ in submodel.named_parameters()]}")
    
    def init_finished(self):
        return True
    
    def protocol(self):
        self.print("step0: convert_fsdp/megatron_inplace")
        if self.engine_type == "fsdp":
            state_dict, sharding = convert_fsdp_inplace("fsdp", self.model)
        elif self.engine_type == "megatron":
            parameter_mapping = create_parameter_mapping("Megatron", QWEN_MODEL_PATH)
            state_dict, sharding = convert_megatron_inplace(parameter_mapping, self.model)
        # self.print(f"state_dict keys: {state_dict.keys()}")
        self.state_dict = state_dict
        self.sharding = sharding
        self.state_dict_keys = list(state_dict.keys())
        self.unified_sharding = None
        self.print("step1: connect_to_server")
        self.client.connect_to_server()
        self.print("step2: send_local_sharding")
        self.client.send_local_sharding(sharding)
        self.print("step3: wait_for_server_sharding")
        self.unified_sharding = self.client.wait_for_server_sharding()
        self.print("step4: register_local_tensors")
        self.client.register_local_tensors(self.state_dict, self.unified_sharding)
        self.print("step5: send_local_info")
        self.client.send_local_info()
        self.print("step6: wait_for_server_info (client infos & comm plan)")
        self.client.wait_for_server_info()
        self.print("step7: send_local_temp_mapping")
        self.client.send_local_temp_mapping()
        self.print("step8: wait_for_server_temp_mappings")
        self.client.wait_for_server_temp_mappings()
        self.print("protocol done.")

    def push_to_ps(self, ps_agent_name, ps_client_name):
        for key in self.state_dict_keys:
            # self.print(f"push {key} to {ps_client_name}")
            self.client.client_write(ps_agent_name, ps_client_name, key, b"train_push")
            # self.client.wait(key, b"train_push", "WRITE", target_client=ps_client_name)
        return True
    
    def wait_push_done(self, ps_client_name):
        futures = []
        for key in self.state_dict_keys:
            self.client.wait(key, b"train_push", "WRITE", target_client=ps_client_name)
            futures.append(self.ps_for_push_worker_handles[ps_client_name].transfer_train_to_gen.remote(key))
        ray.get(futures[-1])
        return True
    
    def shutdown(self):
        self.client.shutdown()
        dist.destroy_process_group()

@ray.remote(num_cpus=1)
class GenClientActor:
    def __init__(self, rank, world_size, server_name, psrl_config, backend, torch_port, log_dir, nixl_interface: NIXLInterface):
        from vllm import LLM
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(torch_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        self.client_name = f"gen_client_{rank}"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.client_name}.log")
        self.print = make_dual_print(log_path, prefix=self.client_name)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.manual_seed(42)
        os.environ["LOCAL_RANK"] = str(0)
        self.print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        assert world_size % 4 == 0, f"world_size {world_size} is not divisible by 4"
        llm = LLM(
            model=QWEN_MODEL_PATH,
            dtype="bfloat16",
            tensor_parallel_size=world_size // 2,
            pipeline_parallel_size=world_size // 2,
            distributed_executor_backend="external_launcher",
            enforce_eager=True,
            disable_custom_all_reduce=True,
            # disable_mm_preprocessor_cache=True,
            seed=42,
        )
        self.model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        self.print(f"local rank {rank} model: {self.model}")
        '''
        seen_module_prefixes = set()
        for module_prefix, module in self.model.named_modules():
            seen_module_prefixes.add(module_prefix)
        self.print(f"seen_module_prefixes: {seen_module_prefixes}, has lm_head: {'lm_head' in seen_module_prefixes}, has lm_head module: {hasattr(self.model, 'lm_head')}")
        '''
        self.rank = rank
        self.tp_rank = rank % 2
        self.client = NIXLStorageClient(
            client_name=self.client_name,
            server_name=server_name,
            use_gpu=True,
            client_type=NIXLClientType.PULL_SIDE,
            nixl_config=psrl_config.nixl,
            nixl_interface=nixl_interface
        )
        
    def init_finished(self):
        return True
    
    def get_state_dict(self):
        return {k: v.cpu() for k, v in self.state_dict.items()}

    def protocol(self):
        self.print("step0: convert_vllm_inplace")
        param_mapping = create_parameter_mapping(type(self.model), QWEN_MODEL_PATH)
        state_dict, sharding = convert_vllm_inplace(param_mapping, self.model, tp_rank=self.tp_rank)
        # self.print(f"state_dict keys: {list(state_dict.keys())}")
        self.state_dict = state_dict
        self.sharding = sharding
        self.state_dict_keys = list(state_dict.keys())
        self.unified_sharding = None
        self.print("step1: connect_to_server")
        self.client.connect_to_server()
        self.print("step2: send_local_sharding")
        self.client.send_local_sharding(sharding)
        self.print("step3: wait_for_server_sharding")
        self.unified_sharding = self.client.wait_for_server_sharding()
        # self.print(f"unified_sharding: {self.unified_sharding}")
        self.print("step4: register_local_tensors")
        self.client.register_local_tensors(self.state_dict, self.unified_sharding)
        self.print("step5: send_local_info")
        self.client.send_local_info()
        self.print("step6: wait_for_server_info (client infos & comm plan)")
        self.client.wait_for_server_info()
        # self.print(f"comm plan: {self.client._comm_plan}")
        self.print("step7: send_local_temp_mapping")
        self.client.send_local_temp_mapping()
        self.print("step8: wait_for_server_temp_mappings")
        self.client.wait_for_server_temp_mappings()
        self.print("protocol done.")

    def pull_from_ps(self, ps_agent_name, ps_client_name):
        for key in self.state_dict_keys:
            # self.print(f"pull {key} from {ps_client_name}")
            self.client.client_read(ps_agent_name, ps_client_name, key, b"gen_pull")
            # self.client.wait(key, b"gen_pull", "READ", target_client=ps_client_name)
        return True
    
    def wait_pull_done(self, ps_client_name):
        for key in self.state_dict_keys:
            self.client.wait(key, b"gen_pull", "READ", target_client=ps_client_name)
        return True
    
    def shutdown(self):
        self.client.shutdown()

def create_ps_worker_group(train_engine_type, num_ps, psrl_config, model_path, nixl_interface: NIXLInterface):
    model_config = OmegaConf.create({
        "path": model_path, "use_shm": False, "trust_remote_code": False
    })
    ray_nodes = ray.nodes()
    nodes = [ray_nodes[i] for i in range(num_ps)]
    ps_resource_pool = PSResourcePool([
        PSResourceSpec(node_ip=node["NodeManagerAddress"], node_id=node["NodeID"], attached_gpu_id=None)
        for node in nodes
    ])
    storage_plan = PSStoragePlan(
        train_model_dtype=torch.float32 if train_engine_type == "fsdp" else torch.bfloat16,
        gen_model_dtype=torch.bfloat16
    )
    ps_cls_with_init = PSClassWithInitArgs(ray.remote(PSStorageWorker), storage_plan, model_config, psrl_config, nixl_interface)
    ps_wg = PSWorkerGroup(ps_resource_pool, ps_cls_with_init)
    return ps_wg

def test_nixl_e2e():
    log_dir = os.environ.get("PSRL_WORKSPACE") + "/psrl/unit_tests/nixl/log"
    os.makedirs(log_dir, exist_ok=True)
    ray.init(ignore_reinit_error=True)
    listen_ip = os.environ.get("LOCAL_IP")
    listen_port = 23459
    server_name = GLOBAL_META_SERVER_NAME
    backend = "nccl"
    torch_port_train = 29502
    torch_port_gen = 29503
    num_train = 8
    num_gen = 4
    num_ps = 2
    train_engine_type = "megatron"
    
    psrl_config = OmegaConf.create({
        "logging_path": log_dir,
        "ps_manager_ip": listen_ip,
        "nixl": {
            "server_mode": "meta_server",
            "server_ip": listen_ip,
            "server_port": listen_port,
            "max_pinned_temp_memory_slots": 4
        },
        "ps_mode": "nixl_cpu"
    })
    
    nixl_interface = NIXLInterface(
        port_scanner=GLOBAL_PORT_SCANNER
    )
    # nixl_interface = NIXLInterface()
    
    start_time = time.time()
    ip_to_node_id = {node['NodeManagerAddress']: node['NodeID'] for node in ray.nodes()}
    assert listen_ip in ip_to_node_id, f"listen_ip {listen_ip} not found in ray nodes"
    server = MetaServerActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=ip_to_node_id[listen_ip],
            soft=False
        )
    ).remote(server_name, psrl_config, num_train + num_gen + num_ps, log_dir)
    ray.get(server.init_finished.remote())
    end_time = time.time()
    print(f"[PASS] server init done. time: {end_time - start_time}s")
    
    start_time = time.time()
    ps_wg = create_ps_worker_group(train_engine_type, num_ps, psrl_config, QWEN_MODEL_PATH, nixl_interface)
    ray.get(ps_wg.execute_all_async("init_model"))
    ray.get(ps_wg.execute_all_async("init_nixl_client"))
    ps_agent_names = ray.get(ps_wg.execute_all_async("get_nixl_agent_name"))
    ps_for_push_names = ray.get(ps_wg.execute_all_async("get_nixl_train_storage_client_name"))
    ps_for_pull_names = ray.get(ps_wg.execute_all_async("get_nixl_gen_storage_client_name"))
    print(f"ps_for_push_names: {ps_for_push_names}")
    print(f"ps_for_pull_names: {ps_for_pull_names}")
    ps_for_push_worker_handles = {}
    for ps_for_push_name in ps_for_push_names:
        ps_for_push_worker_handles[ps_for_push_name] = ps_wg.distinguish_worker_by_method(lambda worker: ray.get(worker.get_nixl_train_storage_client_name.remote()) == ps_for_push_name)
    end_time = time.time()
    print(f"[PASS] ps init done. time: {end_time - start_time}s")
    
    # train_pg = placement_group([{"CPU": 1, "GPU": 1} for _ in range(num_train)], strategy="PACK")
    # gen_pg = placement_group([{"CPU": 1, "GPU": 1} for _ in range(num_gen)], strategy="PACK")
    # ray.get(train_pg.ready())
    # ray.get(gen_pg.ready())
    
    start_time = time.time()
    train_actors = [
        TrainClientActor.options(
            num_gpus=1,
            # scheduling_strategy=PlacementGroupSchedulingStrategy(train_pg, placement_group_bundle_index=rank)
        ).remote(train_engine_type, rank, num_train, server_name, psrl_config, backend, torch_port_train, log_dir, nixl_interface, ps_for_push_worker_handles)
        for rank in range(num_train)
    ]
    gen_actors = [
        GenClientActor.options(
            num_gpus=1,
            # scheduling_strategy=PlacementGroupSchedulingStrategy(gen_pg, placement_group_bundle_index=rank)
        ).remote(rank, num_gen, server_name, psrl_config, backend, torch_port_gen, log_dir, nixl_interface)
        for rank in range(num_gen)
    ]
    for i in range(num_train):
        ray.get(train_actors[i].init_finished.remote())
    for i in range(num_gen):
        ray.get(gen_actors[i].init_finished.remote())
    end_time = time.time()
    print(f"[PASS] client init done. time: {end_time - start_time}s")

    start_time = time.time()
    # Run protocol for server, clients, and PS workers
    server.protocol.remote()
    futures = []
    for t in train_actors:
        futures.append(t.protocol.remote())
    for g in gen_actors:
        futures.append(g.protocol.remote())
    futures.extend(ps_wg.execute_all_async("nixl_protocol"))
    ray.get(futures)
    end_time = time.time()
    print(f"[PASS] protocol done. time: {end_time - start_time}s")

    # Each train client pushes to each PS worker
    start_time = time.time()
    futures = []
    for ps_agent_name, ps_for_push_name in zip(ps_agent_names, ps_for_push_names):
        for t in train_actors:
            t.push_to_ps.remote(ps_agent_name, ps_for_push_name)
        for t in train_actors: 
            futures.append(t.wait_push_done.remote(ps_for_push_name))
    ray.get(futures)
    end_time = time.time()
    print(f"[PASS] train push to all ps done. time: {end_time - start_time}s")

    # Each gen client pulls from each PS worker
    start_time = time.time()
    futures = []
    for ps_agent_name, ps_for_pull_name in zip(ps_agent_names, ps_for_pull_names):
        for g in gen_actors:
            g.pull_from_ps.remote(ps_agent_name, ps_for_pull_name)
        for g in gen_actors:
            futures.append(g.wait_pull_done.remote(ps_for_pull_name))
    ray.get(futures)
    end_time = time.time()
    print(f"[PASS] gen pull from all ps done. time: {end_time - start_time}s")

    # Fetch and verify gen client state_dicts
    print("[CHECK] Verifying GenClientActor state_dicts are all ones...")
    gen_state_dicts = ray.get([g.get_state_dict.remote() for g in gen_actors])
    for gen_idx, state_dict in enumerate(gen_state_dicts):
        for k, v in state_dict.items():
            assert torch.allclose(v, torch.ones_like(v), atol=1e-6), f"[VERIFY] Gen client {gen_idx} param {k} is not all ones: {v}"
    print("[PASS] All GenClientActor parameters are all ones.")

    # Shutdown all actors and Ray
    for t in train_actors:
        ray.get(t.shutdown.remote())
    for g in gen_actors:
        ray.get(g.shutdown.remote())
    ray.get(ps_wg.execute_all_async("shutdown"))
    ray.get(server.shutdown.remote())
    ray.shutdown()

if __name__ == "__main__":
    test_nixl_e2e()