python plot.py ../precision_test/experimental/megatron_psrl_log/staleness_3_tp_bl \
--substring StatCollector \
--xlabel "time(s)" \
--ylabel "generation throughput" \
--out instance_throughput.png \
--mode subplot \
--processor instance_throughput_indexed_by_time