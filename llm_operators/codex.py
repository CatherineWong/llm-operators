"""
codex.py
Utilities that call a large language-code model.
"""

import csv
import os
import random
import time
import json
from collections import Counter, defaultdict
import llm_operators.experiment_utils as experiment_utils

import openai
from openai.error import APIError, APIConnectionError, InvalidRequestError, RateLimitError, ServiceUnavailableError, Timeout

from llm_operators.pddl import PDDLPlan

# TODO(Jiayuan Mao @ 2023/02/04): use a principled way to control the random seed.
random.seed(0)

NONE = "NONE"
STOP_TOKEN = "\n<END>\n"
OPERATOR_SAMPLING_START_TOKEN = "<START>"
OPERATOR_SAMPLING_END_TOKEN = "<END>"
OPERATOR_START = ";; Operator: "
EXAMPLE_START = ";; Example: "
NATURAL_LANGUAGE_GOAL_START = ";; Goal: "
COT_GOAL_START = ";; Simplified Goal: "
PDDL_GOAL_START = ";; PDDL Goal: "
PDDL_PLAN_START = ";; PDDL Plan: "
OPERATOR_START_TOKEN = "(:action "
CODEX_PROMPT = "codex_prompt"
CODEX_OUTPUT = "codex_output"
NLgoals_PDDLplans_prompt = "\n;; Natural language goals and PDDL plans\n\n"
REMINDER = ";; Reminder: use ONLY predicates and object types listed in the above PDDL domain. If an English goal contains an object not in the domain, use the most similar available object. All problems are solvable. Propose just ONE goal.\n\n"

DEFAULT_GOAL_TEMPERATURE = 1.0
DEFAULT_OPERATOR_TEMPERATURE = 1.0

COT_OP_START = ";; Parameter Reasoning: We must have ALL objects, receptacles, and tools that would be used to execute the operator as paramaters to the operator."
COT_DICT = {
    # "GotoLocation": "The parameters are the agent, the starting location, and the ending location.",
    # "PickupObjectInReceptacle": "To pickup an object in a receptacle, we interact with the object to be picked up and the receptacle it is in, so both must be parameters.",
    # "PickupObjectNotInReceptacle": "To pickup an object not in a receptacle, we only interact with the object, which must be a parameter.",
    # "PutObjectInReceptacle": "To put an object in a receptacle, we interact with the object and the receptacle that the object will be placed in. So both must be parameters to the operator.",
    "CleanObject": "To clean an object, we interact with the object to be cleaned AND the receptacle that will clean the object (e.g. a sink). So both must be parameters to the operator.",
}


if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Please set this in the shell via `export OPENAI_API_KEY=...`")
openai.api_key = os.environ["OPENAI_API_KEY"]


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
            return e
        except ServiceUnavailableError as e:
            print(e)
            pause_for_rate_limit = True
            completion = e
        except Timeout as e:
            print(e)
            pause_for_rate_limit = True
            completion = e


def get_solved_unsolved_problems_or_supervision(problems):
    raise RuntimeError("get_solved_unsolved_problems_or_supervision should not be called. Use get_solved_unsolved_problems.")

    # keep this for now, but should be removed.
    unsolved_problems = [
        problems[p]
        for p in problems
        if len(problems[p].solved_motion_plan_results) < 1 and not problems[p].should_supervise_pddl_goal
    ]
    solved_problems = [
        problems[p]
        for p in problems
        if (len(problems[p].solved_motion_plan_results) > 0) or problems[p].should_supervise_pddl_goal
    ]
    return unsolved_problems, solved_problems


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


def propose_plans_operators_goals_for_problems(
    current_domain,
    problems,
    supervision_pddl=[],
    n_plan_samples=5,
    n_operator_samples=5,
    temperature=0.0,
    verbose=False,
    output_directory=None,
    command_args=None,
    external_plan_supervision=None,
):
    """
    Proposes PDDL operators, goals, and plans for unsolved problems using Codex.
    Problems are updated with proposed plans and goals.
    ret:
        proposed_codex_operators: PDDL operators proposed by Codex.
    """
    unsolved_problems, solved_problems = get_solved_unsolved_problems(problems, context='pddl_plan')
    if verbose:
        print("Now in: propose_plans_operators_goals_for_problems: ")
        print(f"\t{len(unsolved_problems)} unsolved problems / {len(solved_problems)} solved problems")

    # Condition on: NL goals. Propose: PDDL plans.
    propose_plans_for_problems(
        unsolved_problems=unsolved_problems,
        solved_problems=solved_problems,
        current_domain=current_domain,
        supervision_pddl=supervision_pddl,
        n_samples=n_plan_samples,
        temperature=temperature,
        verbose=verbose,
        output_directory=output_directory,
        experiment_name=command_args.experiment_name,
        use_mock=command_args.debug_mock_propose_plans,
        external_plan_supervision=external_plan_supervision,
    )

    # Condition on: new operator names. Propose: PDDL operator definitions.
    propose_operators_for_problems(
        problems=problems,
        current_domain=current_domain,
        supervision_pddl=supervision_pddl,
        verbose=verbose,
        temperature=temperature,
        n_samples=n_operator_samples,
        output_directory=output_directory,
        initial_pddl_predicates=command_args.initial_pddl_predicates,
        experiment_name=command_args.experiment_name,
        use_mock=command_args.debug_mock_propose_operators,
    )

    # Condition on: NL goals. Propose: PDDL goals.
    propose_goals_for_problems(
        problems=problems,
        current_domain=current_domain,
        output_directory=output_directory,
        supervision_pddl=supervision_pddl,
        verbose=verbose,
        temperature=temperature,
        initial_pddl_predicates=command_args.initial_pddl_predicates,
        experiment_name=command_args.experiment_name,
        use_mock=command_args.debug_mock_propose_goals,
        use_gt=command_args.debug_ground_truth_goals,
    )


def use_ground_truth_operators(current_domain, verbose):
    if verbose:
        print(f"propose_operators_for_problems: using ground truth operators.")
    current_domain.proposed_operators = {
        o: [current_domain.ground_truth_operators[o]]
        for o in current_domain.ground_truth_operators
        if o not in current_domain.operators
    }
    if verbose:
        print("Added the following ground truth operators: ")
        for o in current_domain.proposed_operators:
            print(o)


def propose_plans_operators_for_problems(
    current_domain,
    problems,
    supervision_pddl=[],
    n_plan_samples=5,
    n_operator_samples=5,
    minimum_usage=2,  # Minimum time the operator was used.
    plan_temperature=1.0,
    operator_temperature=DEFAULT_OPERATOR_TEMPERATURE,
    verbose=False,
    output_directory=None,
    command_args=None,
    use_gt=False,
    external_plan_supervision=None,
    external_operator_supervision=None,
    external_operator_sample_with_prompt=True,
    external_operator_names=None,
    resume=False,
    resume_from_iteration=None,
    resume_from_problem_idx=None,
    curr_iteration=None,
    debug_skip_propose_operators_after=None,
    debug_skip_propose_plans_after=None,
):
    unsolved_problems, solved_problems = get_solved_unsolved_problems(problems, context='pddl_plan')
    if use_gt:
        use_ground_truth_operators(current_domain, verbose)
        return

    # Condition on: NL goals. Propose: PDDL plans.
    propose_plans_for_problems(
        unsolved_problems=unsolved_problems,
        solved_problems=solved_problems,
        current_domain=current_domain,
        supervision_pddl=supervision_pddl,
        n_samples=n_plan_samples,
        temperature=plan_temperature,
        verbose=verbose,
        output_directory=output_directory,
        experiment_name=command_args.experiment_name,
        use_mock=command_args.debug_mock_propose_plans,
        external_plan_supervision=external_plan_supervision,
        resume=resume,
        resume_from_iteration=resume_from_iteration,
        resume_from_problem_idx=resume_from_problem_idx,
        curr_iteration=curr_iteration,
        debug_skip_propose_plans_after=debug_skip_propose_plans_after,
    )
    # Condition on: new operator names. Propose: PDDL operator definitions.
    propose_operators_for_problems(
        problems=problems,
        current_domain=current_domain,
        supervision_pddl=supervision_pddl,
        verbose=verbose,
        minimum_usage=minimum_usage,
        temperature=operator_temperature,
        n_samples=n_operator_samples,
        output_directory=output_directory,
        initial_pddl_predicates=command_args.initial_pddl_predicates,
        experiment_name=command_args.experiment_name,
        use_mock=command_args.debug_mock_propose_operators,
        external_operator_supervision=external_operator_supervision,
        external_operator_sample_with_prompt=external_operator_sample_with_prompt,
        external_operator_names=external_operator_names,
        resume=resume,
        resume_from_iteration=resume_from_iteration,
        resume_from_problem_idx=resume_from_problem_idx,
        curr_iteration=curr_iteration,
        debug_skip_propose_operators_after=debug_skip_propose_operators_after,
    )


def propose_predicates_for_problems(problems, current_domain, use_mock):
    # TBD: to be implemented.
    pass


def propose_operators_for_problems(
    problems,
    current_domain,
    supervision_pddl,
    verbose,
    temperature,
    n_samples,
    output_directory,
    initial_pddl_predicates,
    experiment_name,
    use_mock,
    minimum_usage=2,  # Minimum time the operator was used.
    external_operator_supervision=None,
    external_operator_sample_with_prompt=True,
    external_operator_names=None,
    resume=False,
    resume_from_iteration=None,
    resume_from_problem_idx=None,
    curr_iteration=None,
    debug_skip_propose_operators_after=None,
):
    if debug_skip_propose_operators_after >= curr_iteration:
        print(f"debug_skip_propose_operators_after after current iteration, skipping: {curr_iteration}")
        return

    output_json = {}
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"

    # What operators were proposed across the problems? Rank by usage.
    operator_uses, operator_use_counts = get_operator_uses(problems)
    # Propose definitions for any operators we haven't implemented.
    proposed_operators = get_operators_to_propose(
        current_domain, operator_uses, operator_use_counts, minimum_usage, external_operator_names
    )
    output_filepath = f"{experiment_tag}codex_operators_count{'_'.join(initial_pddl_predicates)}.json"
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(operator_use_counts, f)

    output_filepath = f"{experiment_tag}codex_operators{'_'.join(initial_pddl_predicates)}.json"

    if verbose:
        print(f"propose_operators_for_problems:: proposing for {len(proposed_operators)} operators.")
        print(proposed_operators)

    if resume and os.path.exists(os.path.join(output_directory, output_filepath)):
        mock_propose_operators_for_problems(output_filepath, proposed_operators, output_directory, current_domain)

    # Get valid operators, and use a standardized operator mapping.
    if use_mock and experiment_utils.should_use_checkpoint(
        curr_iteration=curr_iteration,
        curr_problem_idx=None,
        resume_from_iteration=resume_from_iteration,
        resume_from_problem_idx=resume_from_problem_idx,
    ):
        try:
            mock_propose_operators_for_problems(output_filepath, proposed_operators, output_directory, current_domain)
            return
        except:
            print("mock for propose_operators_for_problems not found, continuing.")
            pass

    for o in proposed_operators:
        codex_prompt, proposed_operator_definitions = propose_operator_definition(
            current_domain,
            o,
            operator_uses=operator_uses,
            max_operator_examples=10,
            max_usage_examples=10,
            temperature=temperature,
            n_samples=n_samples,
            verbose=verbose,
            initial_pddl_predicates=initial_pddl_predicates,
            supervision_pddl=supervision_pddl,
            external_operator_supervision=external_operator_supervision,
            external_operator_sample_with_prompt=external_operator_sample_with_prompt,
        )
        current_domain.proposed_operators[o] += proposed_operator_definitions
        output_json[o] = {
            CODEX_PROMPT: codex_prompt,
            CODEX_OUTPUT: proposed_operator_definitions,
        }

    if verbose:
        num_proposed = [o for o in proposed_operators if len(current_domain.proposed_operators[o]) > 1]
        print(
            f"\npropose_operators_for_problems: proposed operators for {len(num_proposed)} / {len(proposed_operators)}"
        )
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)


def mock_propose_operators_for_problems(output_filepath, proposed_operators, output_directory, current_domain):
    with open(os.path.join(output_directory, output_filepath), "r") as f:
        output_json = json.load(f)
    print(f"Now in: mock_propose_operators_for_problems: from {os.path.join(output_directory, output_filepath)}")
    for o in proposed_operators:
        if o in output_json:
            current_domain.proposed_operators[o] = output_json[o][CODEX_OUTPUT]
    print(
        f"\tLoaded {len(current_domain.proposed_operators)} mock operators: \n\t"
        + "\n\t".join(current_domain.proposed_operators.keys())
    )


def get_operators_to_propose(
    current_domain, operator_uses, operator_use_counts, minimum_usage, external_operator_names
):
    external_operator_names = [o.lower() for o in external_operator_names] if external_operator_names else []
    existing_operators = set(
        [
            o.lower()
            if o not in current_domain.operator_canonicalization
            else current_domain.operator_canonicalization[o]
            for o in current_domain.operators
        ]
    )

    # Don't match any that have the same characters.
    proposed_operators = [
        p for p in operator_uses if p.lower() not in existing_operators and p.lower() not in external_operator_names
    ]
    # Filter by those with minimum usage.
    proposed_operators = [p for p in proposed_operators if operator_use_counts[p] >= minimum_usage]
    return proposed_operators


def get_operator_uses(problems):
    operator_use_counts = Counter()
    existing_operator_uses = defaultdict(list)
    for problem in problems.values():
        plans = []
        if problem.should_supervise_pddl_plan:
            plans.append(problem.ground_truth_pddl_plan)
        if len(problem.evaluated_pddl_plans) > 0:
            plans.append(problem.get_highest_likelihood_evaluated_pddl_plan())
        if len(problem.proposed_pddl_plans) > 0:
            plans += problem.proposed_pddl_plans
        for plan in plans:
            for action_usage in plan.plan:
                existing_operator_uses[action_usage[PDDLPlan.PDDL_ACTION]].append(action_usage)
                operator_use_counts[action_usage[PDDLPlan.PDDL_ACTION]] += 1
    return existing_operator_uses, operator_use_counts


def get_operator_from_action(action):
    """
    action:
        string of the form (action param1 param2 ..)
    returns:
        the action string (aka operator name)
    """
    tokens = action.strip("()").split(" ")
    op = tokens[0]
    return op


def propose_operator_definition_external_supervision(
    current_domain,
    operator_name_to_define,
    operator_uses,
    temperature=1.0,
    n_samples=4,
    verbose=False,
    external_operator_supervision=None,
    external_operator_sample_with_prompt=True,
):
    from num2words import num2words

    """
    Proposes an operator definition for a given domain, and optionally with examples of operator usages.
    current_domain: an existing PDDL domain.
    operator_uses: dict {operator_name: list of string uses of a given operator in PDDL plans.}
    operator_name_to_define: string name of operator to define.

    :ret: list of up to n_samples operator definitions. Empty list if prompting fails.
    """
    with open(external_operator_supervision + "system.txt") as f:
        system_message = f.read()
    with open(external_operator_supervision + "user.txt") as f:
        sampling_message = f.read()
        OPERATOR_MASK = "<OPERATOR>"
        N_SAMPLES_MASK = "<N_SAMPLES>"
        assert OPERATOR_MASK in sampling_message
        assert N_SAMPLES_MASK in sampling_message
        sampling_message = sampling_message.replace(OPERATOR_MASK, operator_name_to_define)
        sampling_message = sampling_message.replace(N_SAMPLES_MASK, num2words(n_samples))
    codex_prompt = [{"role": "system", "content": system_message}, {"role": "user", "content": sampling_message}]
    try:
        completion = get_completions(
            codex_prompt,
            temperature=temperature,
            n_samples=1,
            max_tokens=1500,
        )[0]
        if not external_operator_sample_with_prompt:
            assert False
        # Parse the tokens out of the completion.
        import re

        operator_matches = re.findall(
            rf"{OPERATOR_SAMPLING_START_TOKEN}(.*?){OPERATOR_SAMPLING_END_TOKEN}", completion, re.DOTALL
        )[:n_samples]

        if verbose:
            print(f"propose_operator_definition:: completion for {operator_name_to_define}")
            for i in range(len(operator_matches)):
                print(f"[{i+1}/{len(operator_matches)}]")
                print(operator_matches[i])
        return codex_prompt, operator_matches
    except:
        return codex_prompt, []


def propose_operator_definition(
    current_domain,
    operator_name_to_define,
    operator_uses={},
    supervision_pddl="",
    max_operator_examples=10,
    max_usage_examples=10,
    temperature=0.3,
    n_samples=3,
    verbose=False,
    initial_pddl_predicates=[],
    external_operator_supervision=None,
    external_operator_sample_with_prompt=True,
):
    """
    Proposes an operator definition for a given domain, and optionally with examples of operator usages.
    current_domain: an existing PDDL domain.
    operator_uses: dict {operator_name: list of string uses of a given operator in PDDL plans.}
    operator_name_to_define: string name of operator to define.

    :ret: list of up to n_samples operator definitions. Empty list if prompting fails.
    """
    if verbose:
        print(f"propose_operator_definition:: operator_name_to_define - {operator_name_to_define}")

    if external_operator_supervision is not None:
        # For now, we also only support sampling with the prompt.
        assert external_operator_sample_with_prompt
        return propose_operator_definition_external_supervision(
            current_domain=current_domain,
            operator_name_to_define=operator_name_to_define,
            operator_uses=operator_uses,
            temperature=temperature,
            n_samples=n_samples,
            verbose=verbose,
            external_operator_supervision=external_operator_supervision,
            external_operator_sample_with_prompt=external_operator_sample_with_prompt,
        )

    else:
        #### TBD: save this entire thing as COT operator examples.
        # Codex prompt header.
        codex_prompt = []
        nl_header = ";;;; Define PDDL planning operators.\n\n"
        codex_prompt.append({"role": "user", "content": nl_header})

        if len(initial_pddl_predicates) <= 0:
            pddl_domain = (
                ";;;; Predicates in the PDDL domain definition.\n"
                + current_domain.domain_definition_to_string(codex_prompt=True)
                + "\n\n"
            )
            translation_header = ";;;; Only use predicates and functions available in the PDDL domain.\n\n"

            codex_prompt.append({"role": "user", "content": pddl_domain + translation_header})

        # Codex prompt exampler operators.
        operator_examples = random.sample(
            list(current_domain.operators.keys()),
            min(len(current_domain.operators), max_operator_examples),
        )
        for o in operator_examples:
            # if o in operator_uses: (ZS 7/28/23 - Remove to allow for more examples)
            operator_str = f"{OPERATOR_START}{o}\n"

            usage_examples = random.sample(
                list(operator_uses[o]),
                min(len(operator_uses[o]), max_usage_examples),
            )
            for use_example in usage_examples:
                operator_str += f"{EXAMPLE_START}{use_example}\n"
            codex_prompt.append({"role": "user", "content": operator_str})

            operator_str = f"{COT_OP_START}\n"
            if o in COT_DICT:
                operator_str += f";;{COT_DICT[o]}\n"
            operator_str += f"{current_domain.operators[o]}\n{STOP_TOKEN}\n"
            codex_prompt.append({"role": "assistant", "content": operator_str})

        # Codex prompt for operator definition.
        operator_str = f"{OPERATOR_START}{operator_name_to_define}\n"
        if operator_name_to_define in operator_uses:
            for use_example in operator_uses[operator_name_to_define]:
                operator_str += f"{EXAMPLE_START}{use_example}\n"
        codex_prompt.append({"role": "user", "content": operator_str})

        try:
            completions = get_completions(
                codex_prompt,
                temperature=temperature,
                stop=STOP_TOKEN,
                n_samples=n_samples,
            )
            if verbose:
                print(f"propose_operator_definition:: completion for {operator_name_to_define}")
                for i in range(len(completions)):
                    print(f"[{i+1}/{len(completions)}]")
                    print(completions[i])
            return codex_prompt, [o for o in completions]
        except Exception as e:
            print(e)
            return codex_prompt, []


def load_external_plan_supervision_strings(external_plan_file):
    with open(external_plan_file) as f:
        all_supervision_json = json.load(f)
    examples_strings = [
        (supervision_json["goal"], get_plan_string_from_supervision_pddl(supervision_json))
        for supervision_json in all_supervision_json
    ]
    return examples_strings


def build_plan_prompt(unsolved_problem, solved_problems, external_plan_file, max_solved_problem_examples=3):
    # Builds a prompt containing external plan examples and a sample set of solved problems.
    if external_plan_file is not None:
        external_plan_strings = load_external_plan_supervision_strings(external_plan_file)
    else:
        external_plan_strings = []
    solved_problem_examples = random.sample(
        solved_problems,
        min(len(solved_problems), max_solved_problem_examples),
    )
    solved_plan_strings = [
        (problem_example.language, get_plan_string_from_solved_problem(problem_example))
        for problem_example in solved_problem_examples
    ]

    all_example_strings = external_plan_strings + solved_plan_strings
    random.shuffle(all_example_strings)

    codex_prompt = ";;;; Given natural language goals, predict a sequence of PDDL actions.\n"
    for goal_language, plan_string in all_example_strings:
        codex_prompt += f"{NATURAL_LANGUAGE_GOAL_START}{goal_language}\n"
        codex_prompt += f"{PDDL_PLAN_START}\n"
        codex_prompt += f"{plan_string}"
        codex_prompt += f"{STOP_TOKEN}\n"
    # Add the current problem.
    codex_prompt += f"{NATURAL_LANGUAGE_GOAL_START}{unsolved_problem.language}\n"
    codex_prompt += f"{PDDL_PLAN_START}\n"
    return codex_prompt


def propose_plans_for_problems(
    unsolved_problems,
    solved_problems,
    current_domain,
    supervision_pddl,
    max_solved_problem_examples=3,
    temperature=0.0,
    n_samples=4,
    verbose=False,
    output_directory=None,
    experiment_name="",
    use_mock=False,
    external_plan_supervision=None,
    resume=False,
    resume_from_iteration=None,
    resume_from_problem_idx=None,
    curr_iteration=None,
    debug_skip_propose_plans_after=None,
):
    """
    Proposes PDDL plans given NL goals.
    Samples from:
    P(pddl_plan | nl_goal, solved pddl_plan+nl_goal pairs)

    unsolved_problems:
        list of Problem objects to be solved
    solved_problems:
        list of Problem objects with solutions
    current_domain:
        Domain object describing the domain
    supervision_pddl:
         If not empty, use these external pddl action sequences. [DEPRECATED]

    external_plan_supervision: file containing external plans.

    Edits the unsolved problem objects - adds plans to the problem.proposed_pddl_plan list
    """
    if debug_skip_propose_plans_after >= curr_iteration:
        print(f"debug_skip_propose_plans_after, skipping this for iteration {curr_iteration}")
        return

    output_json = {}
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"
    output_filepath = f"{experiment_tag}codex_plans.json"
    if resume and os.path.exists(os.path.join(output_directory, output_filepath)):
        mock_propose_plans_for_problems(
            output_filepath,
            unsolved_problems,
            output_directory,
            experiment_name=experiment_name,
        )
        return
    if use_mock and experiment_utils.should_use_checkpoint(
        curr_iteration=curr_iteration,
        curr_problem_idx=None,
        resume_from_iteration=resume_from_iteration,
        resume_from_problem_idx=resume_from_problem_idx,
    ):
        try:
            mock_propose_plans_for_problems(
                output_filepath,
                unsolved_problems,
                output_directory,
                experiment_name=experiment_name,
            )
            return
        except:
            print("mock for propose_plans_for_problems not found, continuing.")
            pass
            import pdb

            pdb.set_trace()

    for idx, unsolved_problem in enumerate(unsolved_problems):
        # Clear out any previous proposed PDDL plans.
        unsolved_problem.proposed_pddl_plans = []

        if verbose:
            print(f"propose_plans_for_problems:: Now on problem {idx} / {len(unsolved_problems)} ... ")
        # Resample a new prompt with new examples for each plan string.
        plan_strings = []
        for _ in range(n_samples):
            codex_prompt = build_plan_prompt(
                unsolved_problem,
                solved_problems,
                external_plan_supervision,
                max_solved_problem_examples=max_solved_problem_examples,
            )
            plan_strings.append(
                get_completions(codex_prompt, temperature=temperature, stop=STOP_TOKEN, n_samples=1)[0]
            )

        for plan_string in plan_strings:
            try:
                plan_string_split = plan_string.split("<END>")[0]
                if verbose:
                    print(unsolved_problem.language + "\n")
                    print(plan_string_split)
                unsolved_problem.proposed_pddl_plans.append(
                    PDDLPlan(plan_string=plan_string_split)
                )  # editing the problem
            except Exception as e:
                print(e)
            continue
        output_json[unsolved_problem.problem_id] = {
            CODEX_PROMPT: codex_prompt,
            CODEX_OUTPUT: plan_strings,
        }

    if verbose:
        num_proposed = [p for p in unsolved_problems if len(p.proposed_pddl_plans) >= 1]
        print(f"\npropose_plans_for_problems:: proposed plans for {len(num_proposed)} / {len(unsolved_problems)}")
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)
    log_proposed_plans_for_problems(
        unsolved_problems,
        output_json,
        output_directory,
        experiment_name=experiment_name,
    )


def mock_propose_plans_for_problems(output_filepath, unsolved_problems, output_directory, experiment_name=""):
    with open(os.path.join(output_directory, output_filepath), "r") as f:
        output_json = json.load(f)
    print(f"mock_propose_plans_for_problems:: from {os.path.join(output_directory, output_filepath)}")
    for unsolved_problem in unsolved_problems:
        if unsolved_problem.problem_id in output_json:
            for plan_string in output_json[unsolved_problem.problem_id][CODEX_OUTPUT]:
                try:
                    plan_string_split = plan_string.split("<END>")[0]
                    unsolved_problem.proposed_pddl_plans.append(
                        PDDLPlan(plan_string=plan_string_split)
                    )  # editing the problem
                except Exception as e:
                    print(e)
                continue
    print(
        f"mock_propose_plans_for_problems:: loaded a total of {len([p for p in unsolved_problems if len(p.proposed_pddl_plans) > 0])} plans for {len(unsolved_problems)} unsolved problems."
    )
    log_proposed_plans_for_problems(unsolved_problems, output_json, output_directory, experiment_name)


def log_proposed_plans_for_problems(unsolved_problems, output_json, output_directory, experiment_name):
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"
    output_filepath = f"{experiment_tag}codex_plans.csv"

    if output_directory:
        print(f"Logging proposed plans: {os.path.join(output_directory, output_filepath)}")
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            fieldnames = [
                "problem",
                "nl_goal",
                "gt_pddl_goal",
                "gt_plan",
                "proposed_plan",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for problem in unsolved_problems:
                for proposed_plan in problem.proposed_pddl_plans:
                    writer.writerow(
                        {
                            "problem": problem.problem_id,
                            "nl_goal": problem.language,
                            "gt_pddl_goal": problem.ground_truth_pddl_problem.ground_truth_goal,
                            "gt_plan": problem.ground_truth_pddl_plan.plan_string,
                            "proposed_plan": proposed_plan.plan_string,
                        }
                    )


def get_plan_string_from_supervision_pddl(supervision_pddl):
    plan = PDDLPlan(plan=supervision_pddl["operator_sequence"])
    return plan.plan_to_string(plan.plan)


def get_plan_string_from_solved_problem(problem):
    """
    problem:
        solved Problem object
    return:
        string to add to the codex input prompt
    """
    if problem.should_supervise_pddl_plan:
        plan = problem.ground_truth_pddl_plan
        return plan.plan_to_string(plan.plan)
    else:
        return problem.get_solved_pddl_plan_string()


def get_domain_string(domain, problem):
    """
    problem:
        PDDL.Problem object
    returns:
        prompt
    """
    domain_string = (
        "<START DOMAIN>\n"
        + domain.domain_for_goal_prompting(problem.ground_truth_pddl_problem.ground_truth_pddl_problem_string)
        + "\n<END DOMAIN>\n\n"
    )
    return "\n".join([REMINDER, domain_string])


def get_solved_goal_prompt(domain, problem):
    """
    problem:
        PDDL.Problem object
    returns:
        prompt
    """
    NL_goal = NATURAL_LANGUAGE_GOAL_START + "\n" + problem.language + "\n"
    COT = COT_GOAL_START + "\n" + problem.chain_of_thought + "\n" if problem.chain_of_thought else ""
    pddl_goal = PDDL_GOAL_START + "\n" + problem.ground_truth_pddl_problem.ground_truth_goal + "\n" + STOP_TOKEN
    return "\n\n".join([NL_goal, COT, pddl_goal])


def get_supervision_goal_prompt(supervision_pddl):
    prompt = ""
    for domain_file in supervision_pddl:
        domain = supervision_pddl[domain_file]["domain"]
        pddl_problem_string = supervision_pddl[domain_file]["pddl_problem_string"]
        domain_string = domain.domain_for_goal_prompting(pddl_problem_string)
        NL_goal = NATURAL_LANGUAGE_GOAL_START + "\n" + supervision_pddl[domain_file]["NL_goal"]
        pddl_goal = PDDL_GOAL_START + "\n" + supervision_pddl[domain_file]["goal_pddl"] + "\n" + STOP_TOKEN
        prompt += "\n\n".join([domain_string, NL_goal, pddl_goal])
    return prompt


def get_unsolved_goal_prompt(domain, problem, include_codex_types=False, include_domain_string=True):
    if include_domain_string:
        domain_string = domain.domain_for_goal_prompting(
            problem.ground_truth_pddl_problem.ground_truth_pddl_problem_string,
            include_codex_types=include_codex_types,
        )
    else:
        domain_string = ""
    NL_goal = "\n" + NATURAL_LANGUAGE_GOAL_START + "\n" + problem.language + "\n"
    return "\n\n".join([domain_string, NL_goal])


def mock_propose_goals_for_problems(output_filepath, unsolved_problems, output_directory, current_domain):
    with open(os.path.join(output_directory, output_filepath), "r") as f:
        output_json = json.load(f)
    print(f"mock_propose_goals_for_problems:: from {os.path.join(output_directory, output_filepath)}")
    for p in unsolved_problems:
        if p.problem_id in output_json:
            p.proposed_pddl_goals.extend(output_json[p.problem_id][CODEX_OUTPUT])
    print(
        f"mock_propose_goals_for_problems:: loaded a total of {len([p for p in unsolved_problems if len(p.proposed_pddl_goals) > 0])} goals for {len(unsolved_problems)} unsolved problems."
    )
    return


def get_custom_codex_prompt(solved_problems):
    """
    Hand selects solved problems to use as prompts for Codex.
    Proof-of-concept until we implement a better heuristic for choosing codex prompts.
    Works on alfred-solvable-200 dataset.
    """
    for i, problem in enumerate(solved_problems):
        print(f"[{i}/{len(solved_problems)}]")
        print(problem.language)
        print(problem.ground_truth_pddl_problem.ground_truth_goal)
        print()

    problem_idxs = [0, 4, 7, 9, 12, 18, 22]

    return [p for i, p in enumerate(solved_problems) if i in problem_idxs]


def propose_goals_for_problems(
    problems,
    current_domain,
    initial_pddl_predicates,
    supervision_pddl,
    experiment_name,
    temperature=2.0,
    include_codex_types=False,
    use_mock=False,
    max_goal_examples=20,
    n_samples=4,
    verbose=False,
    output_directory=None,
    use_gt=False,
    print_every=1,
    args=None,
    resume=False,
    resume_from_iteration=None,
    resume_from_problem_idx=None,
    curr_iteration=None,
):
    random.seed(args.random_seed)

    def get_prompt(max_goal_examples=max_goal_examples):
        # Generate unique prompt for each sample
        prompt = nl_header
        if supervision_pddl:  # Add supervision from external prompts.
            prompt += get_supervision_goal_prompt(supervision_pddl)

        max_goal_examples = min(max_goal_examples, len(solved_problems))
        solved_to_prompt = random.sample(solved_problems, max_goal_examples)

        # domains for all alfred problems should be the same.
        prompt += get_domain_string(current_domain, solved_to_prompt[0])
        for solved_problem in solved_to_prompt:  # constructing the input prompt
            prompt += get_solved_goal_prompt(current_domain, solved_problem)
        prompt += get_unsolved_goal_prompt(
            current_domain,
            problem,
            include_codex_types=include_codex_types,
            include_domain_string=False,
        )
        return prompt

    """
    unsolved_problems:
        list of Problem objects to be solved
    solved_problems:
        list of Problem objects with ground truth plans
    current_domain:
        Domain object describing the domain

    Edits the unsolved problem objects - adds PDDL proposed goals to the problem.proposed_pddl_goals list
    """
    unsolved_problems, solved_problems = get_solved_unsolved_problems(problems, context='pddl_goal')
    if use_gt:
        print("Using ground truth goals, skipping: propose_goals_for_problems")
        return
    output_json = {}
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"
    output_filepath = f"{experiment_tag}codex_goals_{'_'.join(initial_pddl_predicates)}.json"
    if resume and os.path.exists(os.path.join(output_directory, output_filepath)):
        mock_propose_goals_for_problems(output_filepath, unsolved_problems, output_directory, current_domain)
        return
    if use_mock and experiment_utils.should_use_checkpoint(
        curr_iteration=curr_iteration,
        curr_problem_idx=None,
        resume_from_iteration=resume_from_iteration,
        resume_from_problem_idx=resume_from_problem_idx,
    ):
        mock_propose_goals_for_problems(output_filepath, unsolved_problems, output_directory, current_domain)
        return

    if verbose:
        print(f"propose_goals_for_problems:: proposing for {len(unsolved_problems)} unsolved problems.")

    nl_header = "\n;; Natural language goals and PDDL goals\n\n"

    for idx, problem in enumerate(unsolved_problems):
        # For now, we completely reset the goals if we're proposing.
        problem.proposed_pddl_goals = []
        if verbose and idx % print_every == 0:
            print(f"propose_goals_for_problems:: now on {idx} / {len(unsolved_problems)}")
        try:
            goal_strings = []
            for i in range(n_samples):
                prompt = get_prompt()
                goal_strings.append(
                    get_completions(
                        prompt,
                        temperature=temperature,
                        stop=STOP_TOKEN,
                        n_samples=1,
                    )[0]
                )
            output_json[problem.problem_id] = {
                CODEX_PROMPT: prompt,
                CODEX_OUTPUT: goal_strings,
            }
            if verbose:
                print(f'propose_goals_for_problems:: proposed goals for "{problem.language}"::')
                for i, goal_string in enumerate(goal_strings):
                    print(f"[Goal {i+1}/{len(goal_strings)}]")
                    print(goal_string)
            problem.proposed_pddl_goals.extend(goal_strings)  # editing the problem
        except Exception as e:
            print(e)
            continue
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)
