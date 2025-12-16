import logging
import os

import torch


def log_env_info(psrl_logger: logging.Logger, level: int = logging.INFO):
    # Log environment variables
    psrl_logger.log(level, "=== Environment Variables ===")
    for k in sorted(os.environ):
        try:
            psrl_logger.log(level, f"{k}={os.environ[k]}")
        except Exception:
            psrl_logger.log(level, f"{k}=<could not read>")

    # Log PyTorch and CUDA information
    psrl_logger.log(level, "=== PyTorch / CUDA Info ===")
    psrl_logger.log(level, f"torch version: {torch.__version__}")
    psrl_logger.log(level, f"CUDA version (built with PyTorch): {torch.version.cuda}")

    # Check if CUDA is available
    avail = torch.cuda.is_available()
    psrl_logger.log(level, f"torch.cuda.is_available(): {avail}")
    if not avail:
        psrl_logger.log(
            level,
            "CUDA not available — skipping further CUDA device and memory logging.",
        )
        return

    # Number of CUDA devices
    dev_count = torch.cuda.device_count()
    psrl_logger.log(level, f"torch.cuda.device_count(): {dev_count}")

    # Current default CUDA device index
    try:
        cur_dev = torch.cuda.current_device()
        psrl_logger.log(level, f"torch.cuda.current_device(): {cur_dev}")
    except Exception as e:
        psrl_logger.log(level, f"torch.cuda.current_device() failed: {e}")
        cur_dev = None

    # For each device: print properties and memory usage
    for i in range(dev_count):
        # Device properties
        try:
            props = torch.cuda.get_device_properties(i)
            psrl_logger.log(
                level,
                f"Device {i}: name={props.name}, capability={props.major}.{props.minor}, "
                f"total_memory={props.total_memory / 1024**3:.2f} GB",
            )
        except Exception as e:
            psrl_logger.log(level, f"Device {i}: get_device_properties failed: {e}")

        # Memory allocated and reserved by PyTorch
        try:
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            psrl_logger.log(
                level,
                f"Device {i}: memory_allocated={allocated / 1024**2:.2f} MB, "
                f"memory_reserved={reserved / 1024**2:.2f} MB",
            )
        except Exception as e:
            psrl_logger.log(level, f"Device {i}: memory usage query failed: {e}")

        # Free & total memory from CUDA context (cudaMemGetInfo equivalent)
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            psrl_logger.log(
                level,
                f"Device {i}: mem_get_info: free={free_mem / 1024**2:.2f} MB, total={total_mem / 1024**2:.2f} MB",
            )
        except Exception as e:
            psrl_logger.log(level, f"Device {i}: mem_get_info failed: {e}")

        # Current stream / default stream for this device
        try:
            cur_stream = torch.cuda.current_stream(device=i)
            default_stream = torch.cuda.default_stream(device=i)
            psrl_logger.log(
                level,
                f"Device {i}: current_stream={cur_stream}, default_stream={default_stream}",
            )
        except Exception as e:
            psrl_logger.log(level, f"Device {i}: stream query failed: {e}")

    # Optionally: detailed memory summary for current device
    if cur_dev is not None:
        try:
            summary = torch.cuda.memory_summary(device=cur_dev, abbreviated=False)
            psrl_logger.log(level, f"Device {cur_dev}: memory_summary=\n{summary}")
        except Exception as e:
            psrl_logger.log(level, f"Device {cur_dev}: memory_summary failed: {e}")

    psrl_logger.log(level, "=== End of PyTorch / CUDA Info ===")
