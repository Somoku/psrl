import json
from enum import Enum

# Import EventType from external module
try:
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    event_types_path = os.path.join(current_dir, 'static/event_types.json')
    with open(event_types_path, 'r') as f:
        EVENT_TYPES = json.load(f)
except FileNotFoundError:
    print("Event type configuration file not found, please check the path.")

# Create base enum class
class EventTypeBase(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return EVENT_TYPES[name]

# Create enum class
EventType = EventTypeBase('EventType', [(k, v) for k, v in EVENT_TYPES.items()])