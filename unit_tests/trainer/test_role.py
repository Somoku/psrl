from enum import Enum

class Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6

# Dynamically create a subclass with more roles
class PSRL_Role(Role):
    NewRole1 = 7
    NewRole2 = 8

# Usage
print(PSRL_Role.Actor)          # Outputs: PSRL_Role.Actor
print(PSRL_Role.NewRole1)       # Outputs: PSRL_Role.NewRole1

# This is not allowed!
