import asyncio
import inspect
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import psutil
from verl import DataProto

from psrl.utils.reward_score import default_compute_score_async
from psrl.workers.reward.reward_loop import register
from psrl.workers.reward.reward_loop.base import RewardLoopManagerBase


async def single_compute_score(
    evaluation_func,
    completion,
    reference,
    task,
    task_extra_info,
    executor,
    timeout=300.0,
):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        future = loop.run_in_executor(
            executor,
            partial(evaluation_func, task, completion, reference, task_extra_info),
        )
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None  # Default value for failed rows


@register("prime")
class PrimeRewardLoopManager(RewardLoopManagerBase):
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        reward_model_router=None,
        reward_model_tokenizer=None,
    ):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score_async
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_model_router = reward_model_router
        self.reward_model_tokenizer = reward_model_tokenizer

        # PRIME specific config
        self.num_examine = config.reward_model.get("reward_kwargs", {}).get("num_examine", 1)
        self.reward_fn_key = config.reward_model.get("reward_kwargs", {}).get("reward_fn_key", "data_source")
        self.num_processes = config.reward_model.get("reward_kwargs", {}).get("num_processes", 64)
        self.timeout = config.reward_model.get("reward_kwargs", {}).get("timeout", 300.0)

        self.already_print_data_sources = {}

    async def run_single(self, data: DataProto) -> dict:
        """Process a single data item and return reward score.

        Args:
            data: DataProto containing a single data item

        Returns:
            dict: Dictionary containing 'reward_score' and 'reward_extra_info'
        """
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch.get(self.reward_fn_key, data_item.non_tensor_batch.get("data_source"))
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )

        # Use single_compute_score with ProcessPoolExecutor for PRIME's evaluation
        # Check if compute_score is async or sync
        if self.is_async_reward_score:
            # If it's async, call it directly without executor
            try:
                result = await asyncio.wait_for(
                    self.compute_score(data_source, response_str, ground_truth, extra_info),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                print(f"[Timeout] Task timeout: {response_str[:80]}")
                result = None
            except Exception as e:
                print(f"[Error] PRIME scoring failed: {e}, completion: {response_str[:80]}")
                result = None
        else:
            # If it's sync, use ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=1) as executor:
                try:
                    result = await single_compute_score(
                        self.compute_score,
                        response_str,
                        ground_truth,
                        data_source,
                        extra_info,
                        executor,
                        timeout=self.timeout,
                    )
                except Exception as e:
                    print(f"[Error] PRIME scoring failed: {e}, completion: {response_str[:80]}")
                    result = None
                finally:
                    # Clean up processes
                    for pid, proc in executor._processes.items():
                        try:
                            p = psutil.Process(pid)
                            p.terminate()
                            try:
                                p.wait(timeout=5)
                            except psutil.TimeoutExpired:
                                p.kill()
                        except Exception:
                            pass

        reward_extra_info = {}

        # Process result
        score: float
        if result is None or isinstance(result, Exception):
            score = 0.0
        elif isinstance(result, int | float | bool):
            score = float(result)
        elif isinstance(result, dict):
            score = result.get("score", float(result[0]) if result else 0.0)
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = float(result[0]) if result else 0.0

        reward_extra_info["acc"] = score
        reward = score

        # Logging for examination
        if data_source not in self.already_print_data_sources:
            self.already_print_data_sources[data_source] = 0

        if self.already_print_data_sources[data_source] < self.num_examine:
            self.already_print_data_sources[data_source] += 1
            print(f"[PRIME Example] Data source: {data_source}, Response: {response_str[:200]}, Score: {score}")

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
