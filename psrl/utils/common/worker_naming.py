"""
Worker naming utilities for PSRL.

This module is the single source of truth for all worker name construction
logic used across the trainer, PS, gen, and train worker layers. No other
module should construct worker name strings directly.

NIXL name string constants are imported from psrl.utils.common.nixl_names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from psrl.utils.common.nixl_names import (
    NIXL_GEN_CLIENT_PREFIX,
    NIXL_PS_CLIENT_PREFIX,
    NIXL_TRAIN_CLIENT_PREFIX,
)

WorkerRole = Literal["rollout", "actor", "validate"]

# The model a client carries weights for. Several models train at once in PPO, and each
# publishes its own version stream of keys that are *named identically* ("model.layers.0...").
# The model name therefore belongs in the client name: the meta server unifies sharding and
# plans transfers per model, and it knows a client's name before it knows anything else about it.
DEFAULT_MODEL_ROLE = "actor"


@dataclass(frozen=True)
class WorkerKey:
    """
    Structured identifier for a single PSRL worker.

    Usable as a dict key (frozen dataclass generates __hash__ and __eq__).

    Attributes:
        role (WorkerRole): One of 'rollout', 'actor', or 'validate'.
        instance_id (int): 0-based instance index. Actor workers always use 0.
        rank (int): Tensor-parallel rank within the instance.
            For actor workers this is the global rank.
    """

    role: WorkerRole
    instance_id: int
    rank: int

    def __post_init__(self) -> None:
        """Validate field invariants at construction time."""
        if self.role == "actor":
            assert self.instance_id == 0, f"Actor workers must have instance_id=0, got: {self.instance_id!r}."

    def to_trainer_name(self) -> str:
        """
        Return the trainer-layer string identifier for this worker.

        This string is used only for logging and debugging. All dict keys
        in ray_trainer.py use WorkerKey directly.

        Returns:
            str: Human-readable name, e.g. 'rollout_I0_R2'.
        """
        return f"{self.role}_I{self.instance_id}_R{self.rank}"

    def to_nixl_client_name(self, n_rollout_instances: int = 0, model_role: str = DEFAULT_MODEL_ROLE) -> str:
        """
        Return the NIXL client name registered with the meta server.

        For validate workers, the NIXL instance_id is offset by
        n_rollout_instances to avoid collision with rollout workers.
        This offset is encapsulated here. Callers must not compute it
        themselves.

        Args:
            n_rollout_instances (int): Total number of rollout instances.
                Only used when role='validate' and ignored for 'rollout' and 'actor'. Defaults to 0.
            model_role (str): Which model this worker carries, e.g. 'actor' or 'critic'.

        Returns:
            str: NIXL client name, e.g. 'NIXLGenClient_I2_R0'.
        """
        if self.role == "rollout":
            return gen_client_name(self.instance_id, self.rank, model_role)
        elif self.role == "validate":
            return gen_client_name(n_rollout_instances + self.instance_id, self.rank, model_role)
        elif self.role == "actor":
            return train_client_name(self.rank, model_role)
        else:
            raise AssertionError(
                f"to_nixl_client_name: unhandled role {self.role!r}. Expected 'rollout', 'actor', or 'validate'."
            )


# --- PS naming helpers ---


def ps_agent_name(rank: int) -> str:
    """
    Return the NIXL agent name for a PS storage worker.

    Args:
        rank (int): PS worker rank.

    Returns:
        str: Agent name, e.g. 'NIXLPSClient_0'.
    """
    return f"{NIXL_PS_CLIENT_PREFIX}_{rank}"


def ps_client_push_name(rank: int) -> str:
    """
    Return the NIXL client name for the push-side PS storage client.

    Args:
        rank (int): PS worker rank.

    Returns:
        str: Push client name, e.g. 'NIXLPSClient_0_for_push'.
    """
    return f"{NIXL_PS_CLIENT_PREFIX}_{rank}_for_push"


def ps_client_pull_name(rank: int) -> str:
    """
    Return the NIXL client name for the pull-side PS storage client.

    Args:
        rank (int): PS worker rank.

    Returns:
        str: Pull client name, e.g. 'NIXLPSClient_0_for_pull'.
    """
    return f"{NIXL_PS_CLIENT_PREFIX}_{rank}_for_pull"


# --- Gen / train worker naming helpers ---


def model_role_of(client_name: str) -> str:
    """
    The model role encoded in a NIXL client name, `DEFAULT_MODEL_ROLE` when the name carries none.

    Args:
        client_name (str): A name built by `gen_client_name` or `train_client_name`.
    """
    for prefix in (NIXL_GEN_CLIENT_PREFIX, NIXL_TRAIN_CLIENT_PREFIX, NIXL_PS_CLIENT_PREFIX):
        if client_name.startswith(prefix + "_"):
            rest = client_name[len(prefix) + 1 :]
            head = rest.split("_", 1)[0]
            # A role is alphabetic; `NIXLTrainClient_3` and `NIXLGenClient_I0_R1` carry none.
            return head if head.isalpha() and not head.startswith("I") else DEFAULT_MODEL_ROLE
    return DEFAULT_MODEL_ROLE


def _role_infix(model_role: str) -> str:
    """`""` for the default role, so existing names and logs are unchanged, `"<role>_"` otherwise."""
    if model_role == DEFAULT_MODEL_ROLE:
        return ""
    assert model_role.isalpha() and not model_role.startswith("I"), (
        f"A model role must be alphabetic and not start with 'I', got {model_role!r}."
    )
    return f"{model_role}_"


def gen_client_name(instance_id: int, rank: int, model_role: str = DEFAULT_MODEL_ROLE) -> str:
    """
    Return the NIXL client name for a gen worker.

    Gen workers receive a pre-computed instance_id that already includes
    any validate offset. They call this helper directly rather than
    constructing a WorkerKey, since they do not know their own role.

    Args:
        instance_id (int): Final NIXL instance_id (offset already applied
            by the caller for validate workers).
        rank (int): Worker rank within the instance.
        model_role (str): Which model this client carries, e.g. 'actor' or 'critic'.

    Returns:
        str: Gen client name, e.g. 'NIXLGenClient_I1_R0' or 'NIXLGenClient_critic_I1_R0'.
    """
    return f"{NIXL_GEN_CLIENT_PREFIX}_{_role_infix(model_role)}I{instance_id}_R{rank}"


def gen_client_instance(client_name: str) -> int:
    """
    Return the instance id encoded in a gen client name made by `gen_client_name`.

    Args:
        client_name (str): Gen client name, e.g. 'NIXLGenClient_I1_R0'.
    """
    prefix = f"{NIXL_GEN_CLIENT_PREFIX}_{_role_infix(model_role_of(client_name))}I"
    assert client_name.startswith(prefix), f"{client_name!r} is not a gen client name."
    return int(client_name[len(prefix) :].split("_R", 1)[0])


def gen_client_rank(client_name: str) -> int:
    """Return the rank encoded in a gen client name made by `gen_client_name`."""
    prefix = f"{NIXL_GEN_CLIENT_PREFIX}_{_role_infix(model_role_of(client_name))}I"
    assert client_name.startswith(prefix), f"{client_name!r} is not a gen client name."
    return int(client_name.rsplit("_R", 1)[1])


def train_client_name(rank: int, model_role: str = DEFAULT_MODEL_ROLE) -> str:
    """
    Return the NIXL client name for a train (actor) worker.

    Args:
        rank (int): Global rank of the train worker.
        model_role (str): Which model this client carries, e.g. 'actor' or 'critic'.

    Returns:
        str: Train client name, e.g. 'NIXLTrainClient_3' or 'NIXLTrainClient_critic_3'.
    """
    return f"{NIXL_TRAIN_CLIENT_PREFIX}_{_role_infix(model_role)}{rank}"
