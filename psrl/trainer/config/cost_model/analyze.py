import json
import argparse

file_path = "./qwen_7b.json"
request_num = 39 + 1
token_num = 269522 + 81 + 19283

with open(file_path, "r") as f:
    data = json.load(f)

print(data)

other_threshold = data["TP2_PP1"]["other_threshold"]
other_latency_b = data["TP2_PP1"]["other_latency_b"]
other_latency_k = data["TP2_PP1"]["other_latency_k"]
attn_latency_b = data["TP2_PP1"]["attn_latency_b"]
attn_latency_k = data["TP2_PP1"]["attn_latency_k"]

latency = max(other_threshold, other_latency_b + other_latency_k * request_num) + attn_latency_b + attn_latency_k * token_num
throughput = request_num / latency

print(f"latency: {latency}")
print(f"throughput: {throughput}")