import random
import os
import json
import llm_operators.experiment_utils as experiment_utils
from llm_operators.codex.common import get_completions
from llm_operators.codex.common import CODEX_PROMPT, CODEX_OUTPUT, STOP_TOKEN
from llm_operators.pddl import PDDLPlan
from collections import Counter, defaultdict

OPERATOR_SAMPLING_START_TOKEN = "<START>"
OPERATOR_SAMPLING_END_TOKEN = "<END>"

EXAMPLE_START = ";; Example: "
OPERATOR_START = ";; Operator: "
OPERATOR_START_TOKEN = "(:action "
DEFAULT_OPERATOR_TEMPERATURE = 1.0

COT_OP_START = ";; Parameter Reasoning: We must have ALL objects, receptacles, and tools that would be used to execute the operator as paramaters to the operator."
COT_DICT = {
    # "GotoLocation": "The parameters are the agent, the starting location, and the ending location.",
    # "PickupObjectInReceptacle": "To pickup an object in a receptacle, we interact with the object to be picked up and the receptacle it is in, so both must be parameters.",
    # "PickupObjectNotInReceptacle": "To pickup an object not in a receptacle, we only interact with the object, which must be a parameter.",
    # "PutObjectInReceptacle": "To put an object in a receptacle, we interact with the object and the receptacle that the object will be placed in. So both must be parameters to the operator.",
    "CleanObject": "To clean an object, we interact with the object to be cleaned AND the receptacle that will clean the object (e.g. a sink). So both must be parameters to the operator.",
}


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
    operator_uses, operator_use_counts = _get_operator_uses(problems)
    # Propose definitions for any operators we haven't implemented.
    proposed_operators = _get_operators_to_propose(
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
        return

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
        codex_prompt, proposed_operator_definitions = _propose_operator_definition(
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


############################################################################################################


def _get_operators_to_propose(
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


def _get_operator_uses(problems):
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


def _get_operator_from_action(action):
    """
    action:
        string of the form (action param1 param2 ..)
    returns:
        the action string (aka operator name)
    """
    tokens = action.strip("()").split(" ")
    op = tokens[0]
    return op


def _propose_operator_definition_external_supervision(
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


def _propose_operator_definition(
    current_domain,
    operator_name_to_define,
    operator_uses={},
    supervision_pddl="",
    max_operator_examples=10,
    max_usage_examples=10,
    max_usage_examples_for_operator_to_define=None,
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

    if max_usage_examples_for_operator_to_define is None:
        max_usage_examples_for_operator_to_define = max_usage_examples

    if external_operator_supervision is not None:
        # For now, we also only support sampling with the prompt.
        assert external_operator_sample_with_prompt
        return _propose_operator_definition_external_supervision(
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
            usage_examples = random.sample(list(operator_uses[operator_name_to_define]), min(len(operator_uses[operator_name_to_define]), max_usage_examples_for_operator_to_define))
            for use_example in usage_examples:
            # for use_example in operator_uses[operator_name_to_define]:
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