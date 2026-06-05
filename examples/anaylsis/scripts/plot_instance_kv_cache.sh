python plot.py /jizhicfs/lhy/psrl_agent/examples/mini_swe/megatron_psrl_log/GRPO-SWE-agent-LM-7B-swe_gym-megatron-staleness_1 \
--substring StatCollector \
--xlabel "time(s)" \
--ylabel "kv cache usage" \
--out new_ablation_ours_kv_cache.png \
--mode subplot \
--processor instance_kv_cache_indexed_by_time
