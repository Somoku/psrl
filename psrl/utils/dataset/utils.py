import logging
import os

import torch

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# AGENT(VERL): copy-in from verl. It should align with verl's implementation.
def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """

    from verl.utils.dataset.rl_dataset import get_dataset_class

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        dataset_cls = get_dataset_class(data_config)
    else:
        from psrl.utils.dataset.rl_dataset import PSRLRLHFDataset

        dataset_cls = PSRLRLHFDataset
        print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset


# AGENT(VERL): copy-in from verl. It should align with verl's implementation.
def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import SequentialSampler

    # torch.utils.data.RandomSampler could not recover properly
    from torchdata.stateful_dataloader.sampler import RandomSampler

    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


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
            data_paths=data_config.file,
            data_config=data_config,
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
