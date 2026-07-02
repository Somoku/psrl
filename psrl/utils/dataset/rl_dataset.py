import copy
import logging
import os

from omegaconf import open_dict
from verl.utils.dataset.rl_dataset import RLHFDataset

from psrl.tools.base import load_all_tools

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRLRLHFDataset(RLHFDataset):
    """RLHFDataset variant that loads PSRL native/function tool schemas.

    The upstream verl dataset performs overlong-prompt filtering during
    ``__init__``.  This wrapper computes PSRL tool schemas before entering the
    parent constructor and injects them when filtering prompts, while keeping the
    rest of verl's dataset implementation unchanged.
    """

    def __init__(self, *args, config, **kwargs):
        self._psrl_tool_schemas = _load_psrl_tool_schemas(config)

        # Current PSRL may depend on a verl version whose RLHFDataset only knows
        # verl-native tool configs.  Prevent that path from trying to import
        # verl tools for PSRL configs; this wrapper supplies schemas instead.
        dataset_config = copy.deepcopy(config)
        with open_dict(dataset_config):
            dataset_config.tool_config_path = None
            dataset_config.function_tool_path = None

        super().__init__(*args, config=dataset_config, **kwargs)
        self.config = config

    def maybe_filter_out_long_prompts(self, dataframe=None):
        if self._psrl_tool_schemas is not None:
            self.tool_schemas = self._psrl_tool_schemas
        return super().maybe_filter_out_long_prompts(dataframe)


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
