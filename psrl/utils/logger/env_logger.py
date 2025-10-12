import logging
import os

def log_env_info(psrl_logger: logging.Logger, level: int = logging.INFO):
    for k in sorted(os.environ):
        psrl_logger.log(level, f"{k}={os.environ[k]}")