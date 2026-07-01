from omegaconf import DictConfig
from verl.utils.config import omega_conf_to_dataclass


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """

    # AGENT(VERL): PSRL distinguish train and rollout gpu resources

    # number of GPUs used in training
    train_n_gpus = config.psrl.deployment.train_ngpus_per_node * config.psrl.deployment.train_nnodes
    if not config.train_actor_rollout_ref.actor.use_dynamic_bsz:
        if config.train_actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                train_n_gpus
                % (model_parallel_size * config.train_actor_rollout_ref.actor.megatron.context_parallel_size)
                == 0
            ), (
                f"n_gpus ({train_n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.train_actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = train_n_gpus // (
                model_parallel_size * config.train_actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = train_n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.train_actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "train_actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "train_actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.train_actor_rollout_ref.actor)
    actor_config.validate(train_n_gpus, config.data.train_batch_size, config.train_actor_rollout_ref.model)

    if not config.train_actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.train_actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.train_actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "train_actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.train_actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "train_actor_rollout_ref.rollout",
        )

    if config.algorithm.use_kl_in_reward and config.train_actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(train_n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.train_actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.train_actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    lora_config = config.train_actor_rollout_ref.model.get("lora", {})
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.train_actor_rollout_ref.model.get("lora_rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank > 0 and config.train_actor_rollout_ref.rollout.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    # ---- PSRL specific validation ----

    # Check NIXL compatibility
    if config.psrl.ps_mode == "nixl_cpu" or config.psrl.ps_mode == "nixl_gpu":
        assert config.psrl.nixl.server_ip == config.psrl.ps_manager_ip, (
            "PSManager IP and NIXL server IP must be the same"
        )
        assert config.train_actor_rollout_ref.actor.strategy != "fsdp", (
            "FSDP1 is not supported for NIXL because it uses flat_param"
        )
        print(f"NOTICE: NIXL is enabled. Actor strategy used is {config.train_actor_rollout_ref.actor.strategy}")

    # Check validate mode
    if not config.psrl.colocate_validate_and_train:
        assert config.psrl.fuse_rollout_with_validate, (
            "fuse_rollout_with_validate must be enabled when not colocate_validate_and_train"
        )

    # Check routing strategy
    routing_method = config.psrl.routing_strategy.method
    allowed_routing_methods = {
        "random",
        "round_robin",
        "request_num_balance",
        "throughput_optimal",
        "throughput_optimal_with_budget",
        "throughput_balance",
        "cache_aware",
        "cache_aware_v1",
    }
    assert routing_method in allowed_routing_methods, (
        f"psrl.routing_strategy.method must be one of {sorted(allowed_routing_methods)}, "
        f"got '{routing_method}'. Use 'cache_aware' instead of deprecated 'kv_cache_aware'."
    )

    if routing_method in ("request_num_balance", "throughput_balance"):
        assert config.psrl.status_collection.enable, (
            "status collection must be enabled when using request num balance or throughput balance routing strategy"
        )

    # Check TMS configuration
    if config.psrl.tms.enable_cuda_graph:
        assert config.psrl.tms.range == "all", "TMS CUDA graph can only be enabled when TMS range is 'all'"
    if config.psrl.tms.range not in ["train", "all"]:
        assert (
            config.train_actor_rollout_ref.actor.strategy == "megatron"
            and config.train_actor_rollout_ref.actor.megatron.optimizer_offload
            or config.train_actor_rollout_ref.actor.strategy == "fsdp2"
            and config.train_actor_rollout_ref.actor.fsdp_config.optimizer_offload
        ), "Optimizer offload must be enabled when TMS is not enabled for training workers"

    # Check LMCache and KV transfer configuration
    lmcache_cfg = config.psrl.get("lmcache", {})
    kv_transfer_cfg = config.psrl.routing_strategy.get("kv_transfer", {})

    if kv_transfer_cfg.get("enable", False):
        assert lmcache_cfg.get("enable", False), (
            "psrl.lmcache.enable must be True when psrl.routing_strategy.kv_transfer.enable is True."
        )
        assert lmcache_cfg.get("enable_p2p", False), (
            "psrl.lmcache.enable_p2p must be True when psrl.routing_strategy.kv_transfer.enable is True."
        )
        assert config.psrl.partial_rollout.enable, (
            "psrl.partial_rollout.enable must be True when psrl.routing_strategy.kv_transfer.enable is True."
        )

    if lmcache_cfg.get("enable_p2p", False):
        assert lmcache_cfg.get("enable", False), (
            "psrl.lmcache.enable must be True when psrl.lmcache.enable_p2p is True."
        )
        assert config.psrl.ps_mode in ("nixl_cpu", "nixl_gpu"), (
            "psrl.ps_mode must be nixl_cpu or nixl_gpu when psrl.lmcache.enable_p2p is True "
            "(NIXL infrastructure is required for P2P transfer)."
        )
        '''
        assert not lmcache_cfg.get("clear_on_weight_update", True), (
            "psrl.lmcache.clear_on_weight_update must be False when psrl.lmcache.enable_p2p is True "
            "(LMCache P2PBackend does not support clear; stale entries cannot be flushed on weight update)."
        )
        assert lmcache_cfg.get("multi_version_kv", False), (
            "psrl.lmcache.multi_version_kv must be True when psrl.lmcache.enable_p2p is True "
            "(version tagging is required because P2P KV cannot be cleared on weight sync)."
        )
        '''

    if lmcache_cfg.get("enable", False):
        assert config.gen_actor_rollout_ref.rollout.enable_prefix_caching, (
            "gen_actor_rollout_ref.rollout.enable_prefix_caching must be True "
            "when psrl.lmcache.enable is True (LMCache requires prefix caching)."
        )

    print("[validate_config] All configuration checks passed successfully!")
