import heapq
from collections.abc import Callable, Iterator

from sortedcontainers import SortedDict
from verl import DataProto


def get_priority_by_version(request: DataProto, staleness: int) -> int:
    """Get the priority value for a request based on version tag.

    Args:
        request (DataProto): The request to get priority for.
        staleness (int): The staleness tolerance for version.

    Returns:
        int: Priority value (lower is higher priority).
    """
    assert len(request) == 1, "Request must be a single request"
    if "version_tag" in request.non_tensor_batch:
        if request.non_tensor_batch["version_tag"][0] == -1:
            # Means that the request is not allocated a version tag yet
            # (new request that is not routed yet when enabled dynamic version tag)
            # It should have the lowest priority
            return float("inf")
        return request.non_tensor_batch["version_tag"][0]
    elif "min_version_limit" in request.non_tensor_batch:
        return request.non_tensor_batch["min_version_limit"][0] - staleness
    else:
        raise AssertionError("Request must have either 'version_tag' or 'min_version_limit'")


def get_priority_by_version_and_token_num(
    request: DataProto, staleness: int, short_request_first: bool = False
) -> tuple[int, int, int]:
    """Get the priority value for a request based on version tag and token number.

    Args:
        request (DataProto): The request to get priority for.
        staleness (int): The staleness tolerance for version.
        short_request_first (bool): Whether to prioritize short requests.

    Returns:
        Tuple[int, int, int]: Priority value (lower is higher priority).
    """
    # Prioritize validation requests than training requests
    is_validate = request.meta_info.get("validate", False)
    validate_priority = not is_validate
    version_priority = get_priority_by_version(request, staleness)
    assert "raw_prompt_ids" in request.non_tensor_batch, "raw_prompt_ids is required in non_tensor_batch"
    prompt_token_num = len(request.non_tensor_batch["raw_prompt_ids"][0])
    if "response_unpadded_len" in request.non_tensor_batch:
        response_token_num = request.non_tensor_batch["response_unpadded_len"][0]
    else:
        response_token_num = 0
    if short_request_first:
        token_num_priority = prompt_token_num + response_token_num
    else:
        token_num_priority = -(prompt_token_num + response_token_num)
    return (validate_priority, version_priority, token_num_priority)


class PriorityRequestQueue:
    """A priority queue for routing requests based on version tags."""

    def __init__(self, staleness: int, short_request_first: bool = False):
        """Initialize the priority queue.

        Args:
            staleness (int): The staleness tolerance for version comparison.
        """
        self._queue = []
        self._staleness = staleness
        self._short_request_first = short_request_first
        self._counter = 0  # To ensure FIFO for same priority items

    def put(self, request: DataProto) -> None:
        """Put a request into the priority queue.

        Args:
            request (DataProto): The request to enqueue.
        """
        priority = get_priority_by_version_and_token_num(request, self._staleness, self._short_request_first)
        # Use counter to maintain FIFO order for items with same priority
        heapq.heappush(self._queue, (priority, self._counter, request))
        self._counter += 1

    def pop(self) -> DataProto | None:
        """Pop the highest priority request from the queue.

        Returns:
            Optional[DataProto]: The highest priority request, or None if queue is empty.
        """
        if self._queue:
            _, _, request = heapq.heappop(self._queue)
            return request
        return None

    def peek(self) -> DataProto | None:
        """Peek at the highest priority request without removing it.

        Returns:
            Optional[DataProto]: The highest priority request, or None if queue is empty.
        """
        if self._queue:
            _, _, request = self._queue[0]
            return request
        return None

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if queue is empty, False otherwise.
        """
        return len(self._queue) == 0

    def size(self) -> int:
        """Get the current size of the queue.

        Returns:
            int: Number of items in the queue.
        """
        return len(self._queue)

    def iter_priority(self) -> Iterator[DataProto]:
        """Iterate over all requests in priority order (highest priority first).

        Yields:
            DataProto: Requests in priority order.

        Example:
            for request in queue.iter_priority():
                print(request)
        """
        # Create a copy of the queue to avoid modifying the original
        queue_copy = self._queue.copy()
        # Sort by priority (heapq maintains heap property, but we need sorted order)
        sorted_queue = sorted(queue_copy, key=lambda x: (x[0], x[1]))
        for _, _, request in sorted_queue:
            yield request

    def filter_by_condition(
        self, condition: Callable[[DataProto], bool], guarantee_order: bool = True
    ) -> list[DataProto]:
        """Filter requests by a condition and return them sorted by priority.

        Args:
            condition (Callable[[DataProto], bool]): A function that takes a request
                and returns True if it should be included in the result.
            guarantee_order (bool): Whether to guarantee the order of the filtered requests.

        Returns:
            List[DataProto]: Filtered requests sorted by priority (highest priority first).

        Example:
            # Filter requests with version_tag == 5
            filtered = queue.filter_by_condition(
                lambda req: req.non_tensor_batch.get("version_tag", [None])[0] == 5
            )
        """
        # Filter and collect matching requests with their priorities
        matching_requests = []
        for priority, counter, request in self._queue:
            if condition(request):
                matching_requests.append((priority, counter, request))

        if guarantee_order:
            # Sort by priority (and counter for FIFO within same priority)
            matching_requests.sort(key=lambda x: (x[0], x[1]))

        # Return only the requests (without priority and counter)
        return [request for _, _, request in matching_requests]


class MultiPriorityRequestQueue:
    """Multiple priority queues for routing requests based on version tags.

    This class maintains multiple PriorityRequestQueue instances, each identified by a queue ID.
    Requests are routed to queues based on a provided selector function.
    """

    def __init__(
        self,
        staleness: int,
        queue_selector: Callable[[DataProto, int], int] = get_priority_by_version,
        short_request_first: bool = False,
    ) -> None:
        """Initialize the multi-priority queue.

        Args:
            staleness (int): The staleness tolerance for version comparison.
            queue_selector (Callable[[DataProto, int], int]): A function that takes a request and staleness
                and returns the queue ID (int) to which the request should be routed.
            short_request_first (bool): Whether to prioritize short requests.
        """
        self._staleness = staleness
        self._queue_selector = queue_selector
        self._short_request_first = short_request_first
        self._queues = SortedDict(lambda x: x)

    def put(self, request: DataProto) -> None:
        """Put a request into the appropriate priority queue.

        The queue is selected based on the queue_selector function. If the selected
        queue doesn't exist, it will be created automatically.

        Args:
            request (DataProto): The request to enqueue.
        """
        queue_id = self._queue_selector(request, self._staleness)
        if queue_id not in self._queues:
            self._queues[queue_id] = PriorityRequestQueue(self._staleness, self._short_request_first)
        self._queues[queue_id].put(request)

    def get_queue(self, queue_id: int) -> PriorityRequestQueue | None:
        """Get the priority queue for a given queue ID.

        Args:
            queue_id (int): The ID of the queue to retrieve.

        Returns:
            Optional[PriorityRequestQueue]: The queue if it exists, None otherwise.
        """
        return self._queues.get(queue_id)

    def get_first_queue(self) -> PriorityRequestQueue:
        """Get the first priority queue.

        Returns:
            PriorityRequestQueue: The first queue.
        """
        assert len(self._queues) > 0, "There are no queues in the multi-priority queue"
        return list(self._queues.values())[0]

    def remove_queue(self, queue_id: int) -> bool:
        """Remove a queue by its ID.

        Args:
            queue_id (int): The ID of the queue to remove.

        Returns:
            bool: True if the queue was removed, False if it didn't exist.
        """
        if queue_id in self._queues:
            del self._queues[queue_id]
            return True
        return False

    def queue_ids(self) -> list[int]:
        """Get all queue IDs.

        Returns:
            List[int]: A list of all queue IDs.
        """
        return list(self._queues.keys())

    def iter_queues(self) -> Iterator[tuple[int, PriorityRequestQueue]]:
        """Iterate over all queues.

        Yields:
            Tuple[int, PriorityRequestQueue]: (queue_id, queue) pairs.

        Example:
            for queue_id, queue in multi_queue.iter_queues():
                print(f"Queue {queue_id} has {queue.size()} items")
        """
        yield from self._queues.items()

    def iter_all_requests(self, queue_order: list[int] | None = None) -> Iterator[tuple[int, DataProto]]:
        """Iterate over all requests from all queues.

        Args:
            queue_order (Optional[List[int]]): Optional list of queue IDs specifying
                the order to iterate queues. If None, queues are iterated in
                arbitrary order.

        Yields:
            Tuple[int, DataProto]: (queue_id, request) pairs.

        Example:
            for queue_id, request in multi_queue.iter_all_requests():
                print(f"Request from queue {queue_id}")
        """
        if queue_order is None:
            queue_order = list(self._queues.keys())

        for queue_id in queue_order:
            if queue_id in self._queues:
                queue = self._queues[queue_id]
                for request in queue.iter_priority():
                    yield (queue_id, request)

    def iter_requests_by_queue(self, queue_id: int) -> Iterator[DataProto]:
        """Iterate over all requests in a specific queue in priority order.

        Args:
            queue_id (int): The ID of the queue to iterate.

        Yields:
            DataProto: Requests in priority order.

        Raises:
            KeyError: If the queue doesn't exist.

        Example:
            for request in multi_queue.iter_requests_by_queue(0):
                print(request)
        """
        if queue_id not in self._queues:
            raise KeyError(f"Queue {queue_id} does not exist")
        queue = self._queues[queue_id]
        yield from queue.iter_priority()

    def empty(self) -> bool:
        """Check if all queues are empty.

        Returns:
            bool: True if all queues are empty, False otherwise.
        """
        return all(queue.empty() for queue in self._queues.values())

    def size(self) -> int:
        """Get the total number of requests across all queues.

        Returns:
            int: Total number of requests in all queues.
        """
        return sum(queue.size() for queue in self._queues.values())

    def queue_size(self, queue_id: int) -> int:
        """Get the size of a specific queue.

        Args:
            queue_id (int): The ID of the queue.

        Returns:
            int: Number of requests in the queue, or 0 if the queue doesn't exist.
        """
        if queue_id in self._queues:
            return self._queues[queue_id].size()
        return 0

    def queue_empty(self, queue_id: int) -> bool:
        """Check if a specific queue is empty.

        Args:
            queue_id (int): The ID of the queue.

        Returns:
            bool: True if the queue is empty or doesn't exist, False otherwise.
        """
        if queue_id in self._queues:
            return self._queues[queue_id].empty()
        return True

    def remove_empty_queues(self) -> None:
        """Delete all empty queues."""
        for queue_id, queue in self._queues.items():
            if queue.empty():
                self.remove_queue(queue_id)

    def filter_by_condition(self, condition: Callable[[DataProto], bool]) -> list[DataProto]:
        """Filter requests by a condition and return them sorted by priority.

        Args:
            condition (Callable[[DataProto], bool]): A function that takes a request
                and returns True if it should be included in the result.
        """
        return [request for _, request in self.iter_all_requests() if condition(request)]
