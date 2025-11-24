python plot.py ../precision_test/experimental/megatron_psrl_log/staleness_2_tp_bl \
--substring StatCollector \
--xlabel "time(s)" \
--ylabel "instance request num" \
--out instance_request_num.png \
--mode subplot \
--processor instance_request_num_indexed_by_time

python plot.py ../precision_test/experimental/megatron_psrl_log/staleness_2_rq_bl_greedy \
--substring StatCollector \
--xlabel "time(s)" \
--ylabel "instance request num" \
--out instance_request_num_greedy.png \
--mode subplot \
--processor instance_request_num_indexed_by_time