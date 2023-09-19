import random
import os
import json
import csv
import llm_operators.experiment_utils as experiment_utils
from llm_operators.codex.codex_core import get_completions, get_solved_unsolved_problems
from llm_operators.codex.codex_core import CODEX_PROMPT, CODEX_OUTPUT, STOP_TOKEN
from llm_operators.pddl import PDDLPlan
from llm_operators.codex.goal import NATURAL_LANGUAGE_GOAL_START

DEFAULT_PLAN_TEMPERATURE = 1.0
PDDL_PLAN_START = ";; PDDL Plan: "

def propose_plans_for_problems(
    unsolved_problems,
    solved_problems,
    current_domain,
    supervision_pddl,
    max_solved_problem_examples=3,
    n_samples=4,
    temperature=DEFAULT_PLAN_TEMPERATURE,
    external_plan_supervision=None,
    use_mock=False,
    experiment_name="",
    curr_iteration=None,
    output_directory=None,
    resume=False,
    resume_from_iteration=None,
    resume_from_problem_idx=None,
    debug_skip_propose_plans_after=None,
    verbose=False,
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
        mock_propose_plans_for_problems(output_filepath, unsolved_problems, output_directory, experiment_name=experiment_name)
        return
    if use_mock and experiment_utils.should_use_checkpoint(
        curr_iteration=curr_iteration,
        curr_problem_idx=None,
        resume_from_iteration=resume_from_iteration,
        resume_from_problem_idx=resume_from_problem_idx,
    ):
        try:
            mock_propose_plans_for_problems(output_filepath, unsolved_problems, output_directory, experiment_name=experiment_name)
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
            print(f'propose_plans_for_problems:: "{unsolved_problem.language}":')
        # Resample a new prompt with new examples for each plan string.
        plan_strings = []
        for _ in range(n_samples):
            codex_prompt = _build_plan_prompt(unsolved_problem, solved_problems, external_plan_supervision, current_domain, max_solved_problem_examples=max_solved_problem_examples)
            plan_strings.append(get_completions(codex_prompt, temperature=temperature, stop=STOP_TOKEN, n_samples=1)[0])

        for i, plan_string in enumerate(plan_strings):
            try:
                plan_string_split = plan_string.split("<END>")[0]
                if verbose:
                    print(f'[Plan {i} / {len(plan_strings)}]')
                    print(' ', plan_string_split.replace('\n', '; '))
                unsolved_problem.proposed_pddl_plans.append(PDDLPlan(plan_string=plan_string_split))  # editing the problem
            except Exception as e:
                print(e)
            continue
        output_json[unsolved_problem.problem_id] = {
            CODEX_PROMPT: codex_prompt,
            CODEX_OUTPUT: plan_strings,
        }

    if verbose:
        num_proposed = [p for p in unsolved_problems if len(p.proposed_pddl_plans) >= 1]
        print(f"propose_plans_for_problems:: proposed plans for {len(num_proposed)} / {len(unsolved_problems)}")
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
                    unsolved_problem.proposed_pddl_plans.append(PDDLPlan(plan_string=plan_string_split))  # editing the problem
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
            fieldnames = ["problem", "nl_goal", "gt_pddl_goal", "gt_plan", "proposed_plan"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for problem in unsolved_problems:
                for proposed_plan in problem.proposed_pddl_plans:
                    writer.writerow({
                        "problem": problem.problem_id,
                        "nl_goal": problem.language,
                        "gt_pddl_goal": problem.ground_truth_pddl_problem.ground_truth_goal,
                        "gt_plan": problem.ground_truth_pddl_plan.plan_string,
                        "proposed_plan": proposed_plan.plan_string,
                    })
    print('')


############################################################################################################
# Utility functions for composing the prompt for plan proposal.


def _load_external_plan_supervision_strings(external_plan_file):
    with open(external_plan_file) as f:
        all_supervision_json = json.load(f)
    examples_strings = [
        (supervision_json["goal"], _get_plan_string_from_supervision_pddl(supervision_json))
        for supervision_json in all_supervision_json
    ]
    return examples_strings


def _build_plan_prompt(unsolved_problem, solved_problems, external_plan_file, domain, max_solved_problem_examples=3):
    # Builds a prompt containing external plan examples and a sample set of solved problems.
    if external_plan_file is not None:
        external_plan_strings = _load_external_plan_supervision_strings(external_plan_file)
    else:
        external_plan_strings = []
    solved_problem_examples = random.sample(
        solved_problems,
        min(len(solved_problems), max_solved_problem_examples),
    )
    solved_plan_strings = [
        (problem_example.language, _get_plan_string_from_solved_problem(problem_example, domain))
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


def _get_plan_string_from_supervision_pddl(supervision_pddl):
    plan = PDDLPlan(plan=supervision_pddl["operator_sequence"])
    return plan.plan_to_string()


def _get_plan_string_from_solved_problem(problem, domain):
    """
    problem:
        solved Problem object
    return:
        string to add to the codex input prompt
    """
    if problem.should_supervise_pddl_plan:
        plan = problem.ground_truth_pddl_plan
        return plan.plan_to_string()
    else:
        string = problem.get_solved_pddl_plan_string()
        plan = PDDLPlan(plan_string=string)
        return plan.plan_to_string(domain.operator_canonical_name_map)
    
########### Baseline implementation, propose a sequence of completely grounded code policies for each problem given goals and example other problems.
def propose_code_policies_for_problems(
        problems, 
        domain,
        n_samples=4,
        temperature=DEFAULT_PLAN_TEMPERATURE,
        resume=False,
        output_directory=None,
        verbose=False,
        external_code_policies_supervision=None,
        command_args=None
):
    experiment_name = command_args.experiment_name
    unsolved_problems, solved_problems = get_solved_unsolved_problems(problems, context='pddl_plan')
    output_json = {}
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"
    output_filepath = f"{experiment_tag}codex_code_policies.json"

    if resume and os.path.exists(os.path.join(output_directory, output_filepath)):
        mock_propose_code_policies_for_problems(output_filepath, unsolved_problems, output_directory, domain)
        return
    from num2words import num2words
    if verbose:
        print(f"propose_code_policies_for_problems: proposing for {len(unsolved_problems)} unsolved problems.")

    for idx, problem in enumerate(unsolved_problems):
        problem.proposed_code_policies = []
        codex_prompt, proposed_task_predicate_definitions = _propose_task_predicate_definition(domain, solved_problems, problem, n_samples, temperature, external_code_policies_supervision)
        output_json[problem.problem_id] = {
            CODEX_PROMPT: codex_prompt,
            CODEX_OUTPUT: proposed_task_predicate_definitions,
        }
        if verbose:
            print(f'propose_task_predicates_for_problems:: "{problem.language}":')
            for i, goal_string in enumerate(proposed_task_predicate_definitions):
                print(f"[Goal {i+1}/{len(proposed_task_predicate_definitions)}]")
                print(goal_string)
        problem.proposed_code_policies.extend(proposed_task_predicate_definitions)
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)
    

def mock_propose_code_policies_for_problems(output_filepath, unsolved_problems, output_directory, current_domain):
    with open(os.path.join(output_directory, output_filepath), "r") as f:
        output_json = json.load(f)
    print(f"mock_code_policies_for_problems:: from {os.path.join(output_directory, output_filepath)}")
    for p in unsolved_problems:
        if p.problem_id in output_json:
            p.proposed_code_policies.extend(output_json[p.problem_id][CODEX_OUTPUT])
    print(
        f"mock_code_policies_for_problems:: loaded a total of {len([p for p in unsolved_problems if len(p.proposed_code_policies) > 0])} code policies for {len(unsolved_problems)} unsolved problems."
    )
    return

########### Baseline implementation, propose a sequence of completely grounded operator predicates for each problem given goals and example other problems.
def propose_task_predicates_for_problems(
        problems,
        domain,
        n_samples=4,
        temperature=DEFAULT_PLAN_TEMPERATURE,
        resume=False,
        output_directory=None,
        verbose=False,
        external_task_predicates_supervision=None,
        command_args=None
):
    """
    Proposes PDDL task predicates given NL goals.
    """
    experiment_name = command_args.experiment_name
    unsolved_problems, solved_problems = get_solved_unsolved_problems(problems, context='pddl_plan')
    output_json = {}
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"
    output_filepath = f"{experiment_tag}codex_task_predicates.json"

    if resume and os.path.exists(os.path.join(output_directory, output_filepath)):
        mock_propose_task_predicates_for_problems(output_filepath, unsolved_problems, output_directory, domain)
        return

    from num2words import num2words
    if verbose:
        print(f"propose_task_predicates_for_problems:: proposing for {len(unsolved_problems)} unsolved problems.")

    for idx, problem in enumerate(unsolved_problems):
        problem.proposed_pddl_task_predicates = []
        codex_prompt, proposed_task_predicate_definitions = _propose_task_predicate_definition(domain, solved_problems, problem, n_samples, temperature, external_task_predicates_supervision)
        output_json[problem.problem_id] = {
            CODEX_PROMPT: codex_prompt,
            CODEX_OUTPUT: proposed_task_predicate_definitions,
        }
        if verbose:
            print(f'propose_task_predicates_for_problems:: "{problem.language}":')
            for i, goal_string in enumerate(proposed_task_predicate_definitions):
                print(f"[Goal {i+1}/{len(proposed_task_predicate_definitions)}]")
                print(goal_string)
        problem.proposed_pddl_task_predicates.extend(proposed_task_predicate_definitions)
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)
    
def mock_propose_task_predicates_for_problems(output_filepath, unsolved_problems, output_directory, current_domain):
    with open(os.path.join(output_directory, output_filepath), "r") as f:
        output_json = json.load(f)
    print(f"mock_task_predicates_for_problems:: from {os.path.join(output_directory, output_filepath)}")
    for p in unsolved_problems:
        if p.problem_id in output_json:
            p.proposed_pddl_task_predicates.extend(output_json[p.problem_id][CODEX_OUTPUT])
    print(
        f"mock_task_predicates_for_problems:: loaded a total of {len([p for p in unsolved_problems if len(p.proposed_pddl_task_predicates) > 0])} predicate plans for {len(unsolved_problems)} unsolved problems."
    )
    return
############################################################################################################
# Utility functions for composing the prompt for task plan proposal.
def _propose_task_predicate_definition(domain, solved_problems, problem, n_samples, temperature, external_task_predicates_supervision, max_examples=1):
    from num2words import num2words

    with open(external_task_predicates_supervision + "system.txt") as f:
        system_message = f.read()
    
    # We don't add examples from the problems, as we have a limited context. We have examples in the prompt.
    
    with open(external_task_predicates_supervision + "user.txt") as f:
        sampling_message = f.read()
        GOAL, INIT = "<GOAL>", "<INIT>"
        sampling_message = sampling_message.replace(GOAL, problem.language)
        sampling_message = sampling_message.replace(INIT, problem.ground_truth_pddl_problem.ground_truth_init_string)
    codex_prompt = [{"role": "system", "content": system_message}, {"role": "user", "content": sampling_message}]

    TASK_SAMPLING_START_TOKEN = "<START>"
    TASK_SAMPLING_END_TOKEN = "<END>"

    task_predicates = []
    try:
        completions = get_completions(
            codex_prompt,
            temperature=temperature,
            n_samples=n_samples,
            max_tokens=1500,
        )
        for completion in completions:
            # Parse the tokens out of the completion.
            import re

            matches = re.findall(
                rf"{TASK_SAMPLING_START_TOKEN}(.*?){TASK_SAMPLING_END_TOKEN}", completion, re.DOTALL
            )[:1]
            task_predicates += matches
        return codex_prompt, task_predicates
    except:
        return codex_prompt, []
    



        
