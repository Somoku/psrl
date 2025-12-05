from dataclasses import dataclass


@dataclass(frozen=True)
class EntryInfo:
    rollout_instance_id: int
    request_id: int
    model_version: int


# Creating instances of EntryInfo
entry1 = EntryInfo(0, 0, 5)
entry2 = EntryInfo(0, 1, 5)

# Using EntryInfo as dictionary keys
my_dict = {entry1: "some value", entry2: "another value"}

print(my_dict)
