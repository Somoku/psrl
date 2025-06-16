# Deprecated: the `_background_put` has an overhead of serializing the object.
# There is no way to do async put in ray without any serialization/copying overhead.
# We should use `ray.put` directly instead.

import ray
from typing import Any

# A ray remote function for background put
@ray.remote
def _background_put(x: Any):
    return ray.put(x)

# LazyObjectRef stores the handle of the background task
class LazyObjectRef:
    def __init__(self, task_ref):
        self._task_ref = task_ref  # This is a Ray ObjectRef, pointing to the return value of _background_put

    @property
    def task_ref(self):
        return self._task_ref

# Launch the background task
def lazy_put(x: Any) -> LazyObjectRef:
    task_ref = _background_put.remote(x)
    return LazyObjectRef(task_ref)

# Wait for put to finish and get the actual value
def lazy_get(lazy_ref: LazyObjectRef) -> Any:
    object_ref = ray.get(lazy_ref.task_ref)  # Wait for background put to finish, get the object_ref
    return ray.get(object_ref)  # Then get the actual value