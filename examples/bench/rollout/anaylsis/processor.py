import math
from typing import Any, Callable, Dict, Tuple


Processor = Callable[[Dict[str, Any]], Tuple[bool, Dict[str, float], Any]] 
    

def make_intertoken_indexed_processor() -> Processor:
    """
    Return a processor(obj) -> (keep: bool, values: dict, x_value)

    - Keep rows where:
        iteration_stats.num_finished_requests == 0
        AND iteration_stats.time_to_first_tokens_avg == 0.0
    - values: {"inter_token_latencies_avg": float_value}
    - x_value: 1-based index of the kept row (1,2,3,...)
    """
    idx = 0  # enclosed counter

    def processor(obj: Dict[str, Any]) -> Tuple[bool, Dict[str, float], int]:
        nonlocal idx
        it = obj.get("iteration_stats")
        if not isinstance(it, dict):
            return False, {}, None

        # Check num_finished_requests == 0
        try:
            num_finished = it.get("num_finished_requests", None)
            if num_finished is None or int(num_finished) != 0:
                return False, {}, None
        except Exception:
            return False, {}, None

        # Check time_to_first_tokens_avg == 0.0 (exact match)
        try:
            ttf = it.get("time_to_first_tokens_avg", None)
            if ttf is None or float(ttf) != 0.0:
                return False, {}, None
        except Exception:
            return False, {}, None

        # Extract inter_token_latencies_avg as float
        try:
            inter = it.get("inter_token_latencies_avg", None)
            if inter is None:
                return False, {}, None
            val = float(inter)
            if not math.isfinite(val):
                return False, {}, None
        except Exception:
            return False, {}, None

        # Passed all checks -> increment index and return point
        idx += 1
        return True, {"inter_token_latencies_avg": val}, idx

    return processor


def make_prompt_time_indexed_processor() -> Processor:
    """
    Return a processor(obj) -> (keep: bool, values: dict, x_value)
    """
    total_prompt_tokens = 0 
    total_prompt_time = 0

    def processor(obj: Dict[str, Any]) -> Tuple[bool, Dict[str, float], int]:
        nonlocal total_prompt_tokens, total_prompt_time
        it = obj.get("iteration_stats")
        if not isinstance(it, dict):
            return False, {}, None

        # Extract num_prompt_tokens
        num_prompt_tokens = obj["iteration_stats"]["num_prompt_tokens"]
        num_prompt_tokens = float(num_prompt_tokens)
        
        if num_prompt_tokens == 0:
            return False, {}, None
        
        # Extract prompt_throughput
        prompt_throughput = obj["throughput_stats"]["prompt_throughput"]
        prompt_throughput = float(prompt_throughput)
        assert prompt_throughput > 0

        # Passed all checks -> increment index and return point
        total_prompt_tokens += num_prompt_tokens
        total_prompt_time += num_prompt_tokens / prompt_throughput
        print(f"total_prompt_tokens: {total_prompt_tokens}, total_prompt_time: {total_prompt_time}")
        return True, {"prompt_time": num_prompt_tokens / prompt_throughput}, num_prompt_tokens

    return processor


def make_generation_time_indexed_processor() -> Processor:
    """
    Return a processor(obj) -> (keep: bool, values: dict, x_value)
    """
    idx = 0

    def processor(obj: Dict[str, Any]) -> Tuple[bool, Dict[str, float], int]:
        nonlocal idx
        if obj["type"] != "generation_end":
            return False, {}, None

        # Extract generation_time as float
        try:
            generation_time = obj.get("generation_time", None)
            if generation_time is None:
                return False, {}, None
            val = float(generation_time)
            if not math.isfinite(val):
                return False, {}, None
        except Exception:
            return False, {}, None

        # Passed all checks -> increment index and return point
        idx += 1
        return True, {"generation_time": val}, idx

    return processor




