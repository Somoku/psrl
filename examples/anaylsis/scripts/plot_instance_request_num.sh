python plot.py ../paper_exp/e2e/logs/14b_staleness_1_greedy \
--substring StatCollector \
--xlabel "time(s)" \
--ylabel "instance request num" \
--out 14b_instance_request_num_greedy.png \
--mode subplot \
--processor instance_request_num_indexed_by_time

python plot.py ../paper_exp/e2e/logs/32b_staleness_1_greedy \
--substring StatCollector \
--xlabel "time(s)" \
--ylabel "instance request num" \
--out 32b_instance_request_num_greedy.png \
--mode subplot \
--processor instance_request_num_indexed_by_time

python plot.py ../paper_exp/e2e/logs/14b_staleness_3_ours \
--substring StatCollector \
--xlabel "time(s)" \
--ylabel "instance request num" \
--out 14b_instance_request_num_ours.png \
--mode subplot \
--processor instance_request_num_indexed_by_time