"""
task_planner.py
Utilities for generating task level plans.
"""

import os
import time
import os.path as osp
import json
import csv
import pickle
import functools
from typing import Optional, Sequence

from llm_operators.pddl import PDDLPlan
from llm_operators.task_planner_impl import fd_plan_from_strings, pdsketch_onthefly_plan_from_strings
from llm_operators.datasets.dataset_core import Problem
from llm_operators.experiment_utils import run_ipdb

TASK_PLANNER_FD = "task_planner_fd"
TASK_PLANNER_PDSKETCH_ONTHEFLY = "task_planner_pdsketch_onthefly"
TASK_PLANNER_PDSKETCH_ONTHEFLY_HMAX = "task_planner_pdsketch_onthefly_hmax"
TASK_PLANNER_PDSKETCH_ONTHEFLY_HFF = "task_planner_pdsketch_onthefly_hff"


def attempt_task_plan_for_problem(
    pddl_domain,
    problem_idx,
    problem_id,
    problems,
    minimum_n_operators=1,
    random_generator=None,
    use_mock=False,
    command_args=None,
    curr_iteration=0,
    output_directory=None,
    plan_pass_identifier=None,
    plan_attempt_idx=0,
    goal_idx=None,
    resume=False,
    resume_from_iteration=0,
    resume_from_problem_idx=0,
    debug_skip=False,
    debug_proposed_operators: Optional[Sequence[str]] = None,  # Debugging only.
    verbose=False,
):
    """
    Evaluates planner to evaluate task plans for a single planning problems, given a PDDL domain.
    :ret: TRUE if we've added a new PDDL plan for a goal. Updates problem for task plan.
    """
    if verbose:
        print(f"task_planner.attempt_task_plan_for_problem: attempt {problem_idx} / {len(problems)} ID={problem_id} AttemptIdx={plan_attempt_idx} GoalIdx={goal_idx}")
    if debug_skip:
        if verbose:
            print("  ...debug_skip.")
        return False, None

    if command_args.debug_export_failed_pddl:
        # NB(Jiayuan Mao @ 2023/04/07): do a bit of hack here, because we don't have access to "current_iteration" here.
        debug_export_dir = os.path.join(
            command_args.debug_export_failed_pddl,
            osp.basename(output_directory),
            f"problem_{problem_idx}_attempt_{plan_attempt_idx}",
        )
    else:
        debug_export_dir = None

    # experiment_tag = "" if len(command_args.experiment_name) < 1 else f"{command_args.experiment_name}_"
    # output_filepath = f"{experiment_tag}task_plans.json"
    # if use_mock and curr_iteration <= resume_from_iteration and problem_idx <= resume_from_problem_idx:
    #     unsolved_problems = mock_evaluate_task_plans_and_costs_for_problems(
    #         output_filepath, output_directory, problems
    #     )
    #     if problem_id in unsolved_problems or len(problems[problem_id].evaluated_pddl_plans) > 0:
    #         print("Mock found for task plan, continuing...")
    #         any_success = True
    #         new_evaluated_plans = problems[problem_id].evaluated_pddl_plans
    #         return any_success, new_evaluated_plans
    #     else:
    #         print("Mock not found for task plan, continuing...")

    rv = None
    if resume and output_directory is not None:
        rv = mock_task_plan_for_problem_single(problem_id, problems, plan_attempt_idx, goal_idx, output_directory, plan_pass_identifier=plan_pass_identifier)

    if rv is None:
        any_success, new_evaluated_plans, problem_json = sample_task_plans_for_problem(
            pddl_domain=pddl_domain,
            problem=problems[problem_id],
            minimum_n_operators=minimum_n_operators,
            random_generator=random_generator,
            planner_type=command_args.planner,
            command_args=command_args,
            output_directory=output_directory,
            plan_attempt_idx=plan_attempt_idx,
            goal_idx=goal_idx,
            debug_ground_truth_goals=command_args.debug_ground_truth_goals,
            debug_proposed_operators=debug_proposed_operators,
            debug_export_dir=debug_export_dir,
            verbose=verbose,
        )
        if output_directory is not None:
            checkpoint_mock_task_plan_for_problem_single(
                problem_id, problems, plan_attempt_idx, goal_idx, output_directory, plan_pass_identifier=plan_pass_identifier,
                any_success=any_success, evaluated_plans=new_evaluated_plans
            )
    else:
        print('  Using mock task plan result.')
        any_success, new_evaluated_plans = rv
        print(f"  Plan success: {any_success}")
        for value in new_evaluated_plans.values():
            print(f"  Plan string: ", trim_white_spaces(value.plan_string))

    if any_success:
        # Check that this isn't a duplicate of a plan we've already found for that same problem.
        any_success = problems[problem_id].update_evaluated_pddl_plans(new_evaluated_plans)
    return any_success, new_evaluated_plans


def sample_task_plans_for_problem(
    pddl_domain,
    problem,
    planner_type=TASK_PLANNER_FD,
    minimum_n_operators=None,
    random_generator=None,
    command_args=None,
    output_directory=None,
    plan_attempt_idx=0,
    goal_idx=None,
    debug_ground_truth_goals=False,
    debug_proposed_operators: Optional[Sequence[str]] = None,
    debug_export_dir=None,
    verbose=False,
):
    """
    Uses a task_planner to propose samples, so we attempt planning using random subsets of
    proposed_operator set to get a diverse set of plans.

    :ret:
    any_success - whether any of the task plans succeeded.
    all_evaluated_plans: dict(goal : set(plans for this goal))
    overall_problem_json: serializable JSON format.
    """
    overall_problem_json = {"file_name": problem.problem_id, "plans": []}
    any_success = False

    # From the valid operator set, further downsample a set of operators to try, to prevent timeouts and promote operator diversity.
    if debug_proposed_operators:
        sampled_proposed_operators = debug_proposed_operators
    else:
        sampled_proposed_operators = _generate_random_proposed_operator_sample(
            pddl_domain=pddl_domain,
            minimum_n_operators=minimum_n_operators,
            random_generator=random_generator,
        )
    success, evaluated_plans, _ = run_planner(
        pddl_domain=pddl_domain,
        problem=problem,
        goal_idx=goal_idx,
        proposed_operators=sampled_proposed_operators,
        planner_type=planner_type,
        debug_ground_truth_goals=debug_ground_truth_goals,
        debug_export_dir=debug_export_dir,
        verbose=verbose,
    )

    any_success = any_success or success
    for g in evaluated_plans:
        overall_problem_json["plans"].append({"goal": g, "plan": evaluated_plans[g].plan})
    # print(f"Successfully found {len(evaluated_plans)} plans for goals.")

    if output_directory:
        experiment_tag = "" if len(command_args.experiment_name) < 1 else f"{command_args.experiment_name}_"
        output_filepath = f"{experiment_tag}task_planner_results.csv"
        output_filepath = osp.join(output_directory, output_filepath)

        if not osp.exists(output_filepath):
            with open(output_filepath, "w") as f:
                csv.writer(f).writerow(["problem_id", "attempt_id", "minimum_n_operators", "goal", "success", "plan", "sampled_operators"])
        with open(output_filepath, "a") as f:
            writer = csv.writer(f)
            if debug_ground_truth_goals:
                goals = [problem.ground_truth_pddl_problem.ground_truth_goal]
            else:
                goals = problem.proposed_pddl_goals
            for goal in goals:
                writer.writerow([
                    problem.problem_id,
                    plan_attempt_idx,
                    minimum_n_operators,
                    goal,
                    goal in evaluated_plans,
                    evaluated_plans[goal].plan_string if goal in evaluated_plans else None,
                    str(list(sampled_proposed_operators)),
                ])

    return any_success, evaluated_plans, overall_problem_json


def mock_evaluate_task_plans_and_costs_for_problems(output_filepath, output_directory, problems):
    unsolved_problems = set()
    with open(os.path.join(output_directory, output_filepath), "r") as f:
        output_json = json.load(f)
        print(f"Now in: mock_evaluate_task_plans_and_costs_for_problems: from {os.path.join(output_directory, output_filepath)}")
    for plan in output_json:
        if plan["file_name"] in problems:
            problem = problems[plan["file_name"]]
            if len(plan["plans"]) == 0:
                unsolved_problems.add(plan["file_name"])
            else:
                # Set each goal to the set of plans, not a list.
                for plan_json in plan["plans"]:
                    # This updates the evaluated PDDL task plans that succeeded.
                    plan = PDDLPlan(plan=plan_json["plan"])
                    if plan not in set(problem.evaluated_pddl_plans[plan_json["goal"]]):
                        problem.evaluated_pddl_plans[plan_json["goal"]].append(plan)
    print(f"After initialization, there are {len([p for p in problems if len(problems[p].evaluated_pddl_plans) > 0])} problems with plans.")
    return unsolved_problems


@functools.lru_cache()
def get_mocked_task_plan_file(output_directory, plan_pass_identifier):
    filepath = os.path.join(output_directory, f"mocked_task_plans_{plan_pass_identifier}.pkl")
    if not os.path.exists(filepath):
        return dict()
    with open(filepath, "rb") as f:
        return pickle.load(f)


def mock_task_plan_for_problem_single(problem_id, problems, plan_attempt_idx, goal_idx, output_directory, plan_pass_identifier):
    problem: Problem = problems[problem_id]
    goal = sorted(problem.proposed_pddl_goals)[goal_idx]

    mocked_task_plans = get_mocked_task_plan_file(output_directory, plan_pass_identifier)
    identifier = (problem_id, plan_attempt_idx, goal)
    if identifier in mocked_task_plans:
        return mocked_task_plans[identifier]
    return None


def checkpoint_mock_task_plan_for_problem_single(problem_id, problems, plan_attempt_idx, goal_idx, output_directory, plan_pass_identifier, any_success, evaluated_plans):
    problem: Problem = problems[problem_id]
    goal = sorted(problem.proposed_pddl_goals)[goal_idx]

    mocked_task_plans = get_mocked_task_plan_file(output_directory, plan_pass_identifier)
    identifier = (problem_id, plan_attempt_idx, goal)
    mocked_task_plans[identifier] = (any_success, evaluated_plans)
    filepath = os.path.join(output_directory, f"mocked_task_plans_{plan_pass_identifier}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(mocked_task_plans, f)


def _generate_random_proposed_operator_sample(pddl_domain, minimum_n_operators, random_generator, max_attempts=5):
    """
    Samples a set of at least minimum_n_operators operators.
    We choose to include each operator independently based on p(n_operator_successes / n_operator_attempts) in previous trials.
    We make at most max passes through the operator set to do so.
    """
    sampled_operators = set()
    for n_sampling_attempts in range(max_attempts):
        for operator_name in pddl_domain.proposed_operators:
            for operator_body in pddl_domain.proposed_operators[operator_name]:
                (n_operator_successes, n_operator_attempts) = pddl_domain.operators_to_scores[
                    (operator_name, operator_body)
                ]
                # Flip(p) where p(n_operator_successes / n_operator_attempts)
                if random_generator.binomial(1, float(n_operator_successes / n_operator_attempts), 1)[0] > 0:
                    sampled_operators.add(operator_name)
        if len(sampled_operators) >= minimum_n_operators:
            return sampled_operators
    return sampled_operators


def run_planner(
    pddl_domain,
    problem,
    goal_idx=None,
    proposed_operators: Optional[Sequence[str]] = None,
    planner_type=TASK_PLANNER_FD,
    debug_ground_truth_goals=False,
    debug_export_dir=None,
    verbose=False,
):
    """
    pddl_domain: Domain object.
    problem: Problem object.
    planner_type: string indicating which planner to use.
    goal_idx: run planner on a specific goal.

    :ret: Attempts to run planner on each goal in problem.proposed_pddl_goals.
    any_success: whether any of the goals succeeded.
    evaluated_plans : {goal : Plan}
    """
    output_json = {"file_name": problem.problem_id, "plans": []}

    # Get domain strings. Pick the first one that parses
    current_domain_string = pddl_domain.to_string(
        ground_truth_operators=True,
        current_operators=False,
        proposed_operators=proposed_operators,
        show_constants=(not problem.constants_in_problem_file),
    )
    with open('data/domains/alfred_linearized_pdsketch.pddl') as f:
        current_domain_string = f.read()


    if debug_ground_truth_goals:
        sorted_goals = [problem.ground_truth_pddl_problem.ground_truth_goal]
    else:
        sorted_goals = sorted(problem.proposed_pddl_goals)
    if len(sorted_goals) < 1:
        print("\t...no goals, skipping.")

    any_success = False
    evaluated_plans = dict()

    # These must be sorted for this to make sense.
    for current_goal_idx, goal in enumerate(sorted_goals):
        if goal_idx is not None and current_goal_idx != goal_idx:
            continue

        print(f"  Now attempting to plan for goal: {goal_idx} / {len(sorted_goals)}")
        if verbose:
            print(f"    Running planner with existing operators + {len(proposed_operators)} proposed operators: ")
            print(f"    Initial Operators: {pddl_domain.operators.keys()}")
            print(f"    Proposed Operators: {proposed_operators}")
        current_problem_string = problem.ground_truth_pddl_problem.get_pddl_string_with_proposed_goal(
            proposed_goal=goal
        )
        if verbose:
            print("    Language:", problem.language)
            print("    Ground truth goal:", trim_white_spaces(problem.ground_truth_pddl_problem.ground_truth_goal))
            print("    Proposed goal:", trim_white_spaces(goal))
        start_time = time.time()
        if planner_type == TASK_PLANNER_FD:
            success, plan_string = fd_plan_from_strings(
                domain_str=current_domain_string,
                problem_str=current_problem_string,
                verbose=verbose,
            )
            import pdb; pdb.set_trace()
        elif planner_type == TASK_PLANNER_PDSKETCH_ONTHEFLY:
            success, plan_string = pdsketch_onthefly_plan_from_strings(
                domain_str=current_domain_string, problem_str=current_problem_string
            )
        elif planner_type == TASK_PLANNER_PDSKETCH_ONTHEFLY_HMAX:
            success, plan_string = pdsketch_onthefly_plan_from_strings(
                domain_str=current_domain_string,
                problem_str=current_problem_string,
                heuristic="hmax",
            )
        elif planner_type == TASK_PLANNER_PDSKETCH_ONTHEFLY_HFF:
            success, plan_string = pdsketch_onthefly_plan_from_strings(
                domain_str=current_domain_string,
                problem_str=current_problem_string,
                heuristic="hff",
            )
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
        end_time = time.time()
        # Convert the planner into a plan object.
        if verbose:
            print(f"    Plan success: {success}")
            print(f"    Plan string: ", trim_white_spaces(plan_string))
            print(f"    Plan time: {end_time - start_time:.3f}s")
        if success:
            try:
                pddl_plan = PDDLPlan(plan_string=plan_string, pddl_domain=pddl_domain)
                evaluated_plans[goal] = pddl_plan
                output_json["plans"].append({"goal": goal, "plan": pddl_plan.plan})
                any_success = True
            except:
                print(f"    !!!Failed to parse plan string: {plan_string}")
        else:
            if debug_export_dir is not None:
                os.makedirs(debug_export_dir, exist_ok=True)
                with open(osp.join(debug_export_dir, f"goal_{current_goal_idx}_domain.pddl"), "w") as f:
                    f.write(current_domain_string)
                with open(osp.join(debug_export_dir, f"goal_{current_goal_idx}_problem.pddl"), "w") as f:
                    f.write(current_problem_string)
                print(f"    !!!Exported domain and problem to {debug_export_dir}")

    return any_success, evaluated_plans, output_json


def trim_white_spaces(s):
    s = str(s)
    s = s.replace('\n', ' ')
    return " ".join(s.split())
