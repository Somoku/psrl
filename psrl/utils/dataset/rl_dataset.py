import copy
import logging
import os

from omegaconf import OmegaConf, open_dict
from verl.utils.dataset.rl_dataset import RLHFDataset

from psrl.tools.base import load_all_tools

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRLRLHFDataset(RLHFDataset):
    """Load PSRL tool schemas before the parent filters overlong prompts."""

    def __init__(self, *args, config, **kwargs):
        self._psrl_tool_schemas = _load_psrl_tool_schemas(config)

        # The parent understands only veRL-native tool configs, so supply resolved schemas directly.
        dataset_config = copy.deepcopy(config)
        with open_dict(dataset_config):
            dataset_config.tool_config_path = None
            dataset_config.function_tool_path = None

        super().__init__(*args, config=dataset_config, **kwargs)
        self.config = config

    def __getitem__(self, item):
        row = super().__getitem__(item)
        return _attach_reward_model_dicts(row, self.config)

    def maybe_filter_out_long_prompts(self, dataframe=None):
        if self._psrl_tool_schemas is not None:
            self.tool_schemas = self._psrl_tool_schemas
        return super().maybe_filter_out_long_prompts(dataframe)


def _attach_reward_model_dicts(row: dict, config) -> dict:
    """Give the agent loop the reward specification for a dataset row.

    `reward_model_dicts` reaches the reward manager through the rollout request: the agent
    loop selects it from the batch and defaults to `[]` when the dataset does not carry it.
    An empty list resolves to no reward manager, and the score is then the sum over an empty
    list -- zero, indistinguishable from a wrong answer. Standard datasets such as GSM8K
    only ship `reward_model.ground_truth`, so the specification configured on the dataset
    (`data.reward_model_dicts` locally, or per entry in `data.train_datas`) has to be
    attached here.

    A row that already carries the field wins: preprocessed datasets may encode a reward
    specification per sample.
    """
    if row.get("reward_model_dicts") is None:
        configured = config.get("reward_model_dicts", None)
        if configured is not None:
            row["reward_model_dicts"] = OmegaConf.to_container(configured, resolve=True)
    return row


def _load_psrl_tool_schemas(config):
    tool_config_path = config.get("tool_config_path", None)
    function_tool_path = config.get("function_tool_path", None)
    if not tool_config_path and not function_tool_path:
        return None

    try:
        tools = load_all_tools(tool_config_path=tool_config_path, function_tool_path=function_tool_path)
        return [tool.json for tool in tools]
    except Exception as e:
        logger.warning(
            "Failed to initialize PSRL tools for prompt length filtering "
            "(tool_config_path=%s, function_tool_path=%s): %s",
            tool_config_path,
            function_tool_path,
            e,
        )
        return None
