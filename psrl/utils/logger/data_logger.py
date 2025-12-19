import logging

import numpy as np
import torch
from verl.protocol import DataProto


def log_data_protocol(
    inputs: DataProto,
    psrl_logger: logging.Logger,
    log_prefix: str,
    level: int = logging.INFO,
):
    input_tensor_types = {}
    if inputs.batch is not None:
        for k, v in inputs.batch.items():
            input_tensor_types[k] = f"torch.Tensor with dtype {v.dtype} and shape {v.shape}"
    input_non_tensor_types = {}
    for k, v in inputs.non_tensor_batch.items():
        if isinstance(v, torch.Tensor):
            input_non_tensor_types[k] = f"torch.Tensor with dtype {v.dtype} and shape {v.shape}"
        elif isinstance(v, np.ndarray):
            input_non_tensor_types[k] = f"np.ndarray with dtype {v.dtype} and shape {v.shape}"
        elif isinstance(v, list):
            input_non_tensor_types[k] = f"list with length {len(v)}"
        else:
            input_non_tensor_types[k] = f"single value with {type(v)}"

    psrl_logger.log(level, f"[{log_prefix}]: tensor batch have {input_tensor_types}")
    psrl_logger.log(level, f"[{log_prefix}]: non tensor batch have {input_non_tensor_types}")
