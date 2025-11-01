# Modified from verl/utils/reward_score/__init__.py
import asyncio
import inspect
from functools import partial

from verl.utils.import_utils import deprecated


async def default_compute_score_async(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from verl.utils.reward_score import gsm8k
        compute_score_fn = gsm8k.compute_score
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from verl.utils.reward_score import math_reward
        compute_score_fn = math_reward.compute_score
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from verl.utils.reward_score import math_dapo
        compute_score_fn = math_dapo.compute_score
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from verl.utils.reward_score import prime_math
        compute_score_fn = prime_math.compute_score
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from .sandbox_fusion import compute_score as sandbox_fusion_compute_score

            # Pass the URL directly, ground_truth likely contains test cases here
            compute_score_fn = partial(
                sandbox_fusion_compute_score,
                sandbox_fusion_url=sandbox_fusion_url,
                concurrent_semaphore=concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
                continuous=True,
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from verl.utils.reward_score import prime_code

            # Assuming prime_code doesn't need the URL
            compute_score_fn = partial(prime_code.compute_score, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from verl.utils.reward_score import geo3k

        compute_score_fn = geo3k.compute_score
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from verl.utils.reward_score import search_r1_like_qa_em

        compute_score_fn = search_r1_like_qa_em.compute_score

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
    
    is_async_reward_score = inspect.iscoroutinefunction(compute_score_fn)
    if is_async_reward_score:
        try:
            result = await compute_score_fn(
                solution_str,
                ground_truth,
                **kwargs,
            )
        except asyncio.TimeoutError:
            print(f"[Timeout] Task timeout: {solution_str[:80]}")
            result = None
        except Exception as e:
            print(f"[Error] PRIME scoring failed: {e}, completion: {solution_str[:80]}")
            result = None
    else:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: compute_score_fn(
                solution_str,
                ground_truth,
                **kwargs,
            ),
        )
    return result