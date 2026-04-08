import logging
import os

import torch
from verl.trainer.main_ppo import create_rl_dataset

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def create_multi_rl_datasets(
    data_configs,
    tokenizer,
    processor,
):
    """Create multiple datasets for training and validation.

    Arguments:
        data_configs: List of data configs.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.
    Returns:
        datasets (list[Dataset]): A list of datasets for training and validation.
    """
    datasets = []
    for data_config in data_configs:
        dataset = create_rl_dataset(
            data_files=data_config.file,
            config=data_config,
            tokenizer=tokenizer,
            processor=processor,
            max_samples=data_config.get("max_samples", -1),
        )
        datasets.append(dataset)
    return datasets


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids
