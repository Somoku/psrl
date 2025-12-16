import numpy as np
from sklearn.linear_model import LinearRegression

batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# 7B TP1
other_latency = [0.0057, 0.0057, 0.0057, 0.0057, 0.0061, 0.0064, 0.0098, 0.0174, 0.0337]
# 0 kv cache usage
all_init_latency = [
    0.0063,
    0.0063,
    0.0064,
    0.0065,
    0.0068,
    0.0074,
    0.0107,
    0.0189,
    0.0362,
]
# 0.1 kv cache usage
all_final_latency = [None, None, None, None, 0.0118, 0.0129, 0.0156, 0.0233, 0.0397]
tokens = 1415152 * 0.1

# 7B TP2
other_latency = [0.0038, 0.0038, 0.0038, 0.0039, 0.0041, 0.0046, 0.0064, 0.0111, 0.0201]
# 0 kv cache usage
all_init_latency = [
    0.0044,
    0.0044,
    0.0045,
    0.0047,
    0.0049,
    0.0053,
    0.0073,
    0.0123,
    0.0218,
]
# 0.1 kv cache usage
all_final_latency = [None, None, None, None, None, 0.0113, 0.0130, 0.0185, 0.0263]
tokens = 3033664 * 0.1

reg = LinearRegression()
reg.fit(np.array(batch_size[-3:]).reshape(-1, 1), np.array(other_latency[-3:]))
k_other = reg.coef_[0]
b_other = reg.intercept_

print(f"Fitted model: other_latency = {k_other:.6f} * batch_size + {b_other:.6f}")

for i in range(len(batch_size)):
    attn_latency_b = all_init_latency[i] - other_latency[i]
    print(f"batch_size: {batch_size[i]}, attn_latency_b: {attn_latency_b}")
    if all_final_latency[i] is not None:
        attn_latency_k = (all_final_latency[i] - all_init_latency[i]) / tokens
        print(f"batch_size: {batch_size[i]}, attn_latency_k: {attn_latency_k}")
