import random
import os
import json
import csv
import llm_operators.experiment_utils as experiment_utils
from llm_operators.codex.codex_core import get_completions
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
            codex_prompt = _build_plan_prompt(unsolved_problem, solved_problems, external_plan_supervision, max_solved_problem_examples=max_solved_problem_examples)
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


def _build_plan_prompt(unsolved_problem, solved_problems, external_plan_file, max_solved_problem_examples=3):
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
        (problem_example.language, _get_plan_string_from_solved_problem(problem_example))
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
    return plan.plan_to_string(plan.plan)


def _get_plan_string_from_solved_problem(problem):
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
