"""
codex.py
Utilities that call a large language-code model.
"""

import os
import random
import time

import openai
from openai.error import APIError, APIConnectionError, InvalidRequestError, RateLimitError, ServiceUnavailableError, Timeout


# TODO(Jiayuan Mao @ 2023/02/04): use a principled way to control the random seed.
random.seed(0)

NONE = "NONE"
STOP_TOKEN = "\n<END>\n"
CODEX_PROMPT = "codex_prompt"
CODEX_OUTPUT = "codex_output"

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Please set this in the shell via `export OPENAI_API_KEY=...`")
openai.api_key = os.environ["OPENAI_API_KEY"]


def fast_word_count(string):
    return len(string.split())


def get_completions(
    prompt,
    n_samples: int = 1,
    temperature: float = 0.1,
    max_tokens: int = 256,  # Max tokens for completion only.
    engine: str = "gpt-3.5-turbo-16k",  # Add gpt-3.5-turbo-16k, gpt-4-32k, etc
    stop: str = STOP_TOKEN,
    top_p=1,
    logprobs=None,
    max_attempts_rate_limit=5,
    rate_limit_seconds=30,
):
    if get_completions.SKIP_WORD_COUNT:
        total_length = ''
        for prompt_item in prompt:
            total_length += fast_word_count(prompt_item['content'])
        if total_length > 16 * 1000:
            print(f"Warning:: Total length of prompt is {total_length} which is greater than 16k under fast word count.")
            print('Entering an ipdb shell. Type c to continue. This warning will only appear once.')
            print('If you are sure this is the behavior you want, type c to continue and the script will skip this warning in the future.')
            import ipdb; ipdb.set_trace()
            get_completions.SKIP_WORD_COUNT = True

    pause_for_rate_limit = False
    completion = None
    for idx in range(max_attempts_rate_limit):
        if pause_for_rate_limit:
            print(
                f"ERR: Codex rate limit. On attempt {idx}/{max_attempts_rate_limit} after waiting {rate_limit_seconds}s."
            )
            time.sleep(rate_limit_seconds)
            rate_limit_seconds *= 2  # Exponential backoff
        try:
            if engine == "code-davinci-002":
                completion = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature if top_p is None else 1.0,
                    top_p=top_p if temperature is None else 1.0,
                    n=n_samples,
                    stop=stop,
                    frequency_penalty=0,
                    presence_penalty=0,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                )
                return [c["text"] for c in completion["choices"]]
            elif (
                engine == "gpt-3.5-turbo"
                or engine == "gpt-3.5-turbo-16k"
                or engine == "gpt-4-32k"
                or engine == "gpt-4"
            ):
                if type(prompt) != list:
                    prompt = [{"role": "user", "content": prompt}]
                completion = openai.ChatCompletion.create(
                    model=engine,
                    messages=prompt,
                    temperature=temperature if top_p is None else 1.0,
                    top_p=top_p if temperature is None else 1.0,
                    n=n_samples,
                )
                return [c["message"]["content"] for c in completion["choices"]]
            else:
                raise ValueError(f"Engine {engine} not supported.")

        except InvalidRequestError as e:
            print(e)
            return e
        except RateLimitError as e:
            print(e)
            pause_for_rate_limit = True
            completion = e
        except APIConnectionError as e:
            print(e)
            pause_for_rate_limit = True
            completion = e
        except APIError as e:
            print(e)
            pause_for_rate_limit = True
            completion = e
        except ServiceUnavailableError as e:
            print(e)
            pause_for_rate_limit = True
            completion = e
        except Timeout as e:
            print(e)
            pause_for_rate_limit = True
            completion = e


get_completions.SKIP_WORD_COUNT = False


def get_solved_unsolved_problems(problems, context=None):
    if context == 'pddl_goal':
        unsolved_problems = [problems[p] for p in problems if len(problems[p].solved_motion_plan_results) < 1 and not problems[p].should_supervise_pddl_goal]
        solved_problems = [problems[p] for p in problems if (len(problems[p].solved_motion_plan_results) > 0) or problems[p].should_supervise_pddl_goal]
        return unsolved_problems, solved_problems
    elif context == 'pddl_plan':
        unsolved_problems = [problems[p] for p in problems if len(problems[p].solved_motion_plan_results) < 1 and not problems[p].should_supervise_pddl_plan]
        solved_problems = [problems[p] for p in problems if (len(problems[p].solved_motion_plan_results) > 0) or problems[p].should_supervise_pddl_plan]
        return unsolved_problems, solved_problems
    elif context is None:
        unsolved_problems = [problems[p] for p in problems if len(problems[p].solved_motion_plan_results) < 1]
        solved_problems = [problems[p] for p in problems if (len(problems[p].solved_motion_plan_results) > 0)]
        return unsolved_problems, solved_problems
    else:
        raise ValueError("Context must be either 'pddl_goal' or 'pddl_plan' or None.")
