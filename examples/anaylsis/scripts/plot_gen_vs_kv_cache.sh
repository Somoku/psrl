python plot.py ../bench/rollout/7b_exp_wo_attn/details \
--substring TP2 \
--xlabel "kv_cache(%)" \
--ylabel "latency(s)" \
--out gen_vs_kv_cache_wo_attn_tp1.png \
--processor intertoken_indexed_by_kv_cache_usage

python plot.py ../bench/rollout/7b_exp_with_attn/details \
--substring TP2 \
--xlabel "kv_cache(%)" \
--ylabel "latency(s)" \
--out gen_vs_kv_cache_with_attn_tp1.png \
--processor intertoken_indexed_by_kv_cache_usage