"""
motion_planner.py
Utilities for generating motion plans.
"""

import json
import time
import functools
import pickle
import os

from llm_operators.pddl import PDDLPlan


class MotionPlanResult:
    # A motion planning result.
    def __init__(
        self,
        pddl_plan,
        task_success,
        last_failed_operator=None,
        max_satisfied_predicates=None,
        total_trajs_sampled=0,
    ):
        """
        task_success: bool
        last_failed_operator: returns index of last failed operator.
        max_satisfied_predicates: return last failed predicate in that operator.
        """
        self.pddl_plan = pddl_plan  # PDDLPlan
        self.task_success = task_success
        self.last_failed_operator = last_failed_operator
        self.max_satisfied_predicates = max_satisfied_predicates
        self.total_trajs_sampled = total_trajs_sampled

    @classmethod
    def from_json(cls, json):
        return MotionPlanResult(
            pddl_plan=PDDLPlan(plan_string=json["plan"]),
            task_success=json["task_success"],
            last_failed_operator=json["last_failed_operator"],
            max_satisfied_predicates=json["max_satisfied_predicates"],
            total_trajs_sampled=json["total_trajs_sampled"],
        )


def attempt_motion_plan_for_problem(
    pddl_domain,
    problem_idx,
    problem_id,
    problems,
    dataset_name,
    new_task_plans,
    use_mock=False,
    command_args=None,
    curr_iteration=0,
    output_directory=None,
    plan_pass_identifier=None,
    plan_attempt_idx=0,
    resume=False,
    resume_from_iteration=0,
    resume_from_problem_idx=0,
    debug_skip=False,
    verbose=False,
):
    from llm_operators.motion_planner_impl import evaluate_alfred_motion_plans_and_costs_for_goal_plan, evaluate_cw_motion_plans_and_costs_for_goal_plan
    """Attempts to motion plan for a single problem. This attempts the planner on any proposed goals, and any proposed task plans for those goals."""
    if verbose:
        print(f"motion_planner.attempt_motion_plan_for_problem: attempt {problem_idx} / {len(problems)} ID={problem_id} AttemptIdx={plan_attempt_idx}")
    experiment_tag = "" if len(command_args.experiment_name) < 1 else f"{command_args.experiment_name}_"

    # output_filepath = f"{experiment_tag}motion_plans.json"
    # if use_mock and curr_iteration <= resume_from_iteration and problem_idx <= resume_from_problem_idx:
    #     unsolved_problems = mock_evaluate_motion_plans_and_costs_for_problems(output_filepath, output_directory, problems)

    #     # Did we find a solution?
    #     any_success = False
    #     new_motion_plan_keys = []
    #     used_mock = True
    #     if problem_id in unsolved_problems:
    #         print("Mock found for motion plan but no successful motion plan, continuing...")
    #         return any_success, new_motion_plan_keys, used_mock
    #     if len(problems[problem_id].evaluated_motion_planner_results) > 0:
    #         any_success = True
    #         new_motion_plan_keys = problems[problem_id].evaluated_motion_planner_results.keys()
    #         print("Mock found for motion plan, continuing...")
    #         return any_success, new_motion_plan_keys, used_mock
    #     else:
    #         print("Mock not found for motion plan, continuing...")

    any_success = False
    new_motion_plan_keys = []
    used_mock = False
    for pddl_goal, pddl_plan in new_task_plans.items():
        rv = None
        start_time, end_time = 0, 0
        if resume and output_directory is not None:
            rv = mock_motion_plan_for_problem_single(problem_id, pddl_goal, pddl_plan, output_directory, plan_pass_identifier)

        if rv is None:
            start_time = time.time()
            if "alfred" in dataset_name:
                motion_plan_result = evaluate_alfred_motion_plans_and_costs_for_goal_plan(problem_id, problems, pddl_goal, pddl_plan, pddl_domain, motionplan_search_type=command_args.motionplan_search_type, debug_skip=debug_skip, verbose=verbose)
            elif dataset_name == "crafting_world_20230204_mining_only" or dataset_name == "crafting_world_20230829_crafting_only":
                motion_plan_result = evaluate_cw_motion_plans_and_costs_for_goal_plan(problem_id, problems, pddl_goal, pddl_plan, pddl_domain, debug_skip=debug_skip, verbose=verbose)
            else:
                raise ValueError(f'Unknown dataset_name: {dataset_name}.')
            end_time = time.time()
            if output_directory is not None:
                checkpoint_motion_plan_for_problem_single(problem_id, pddl_goal, pddl_plan, output_directory, plan_pass_identifier, motion_plan_result)
        else:
            print('  Using mock motion plan result.')
            motion_plan_result = rv

        new_motion_plan_key = (pddl_goal, motion_plan_result.pddl_plan.plan_string)
        problems[problem_id].evaluated_motion_planner_results[new_motion_plan_key] = motion_plan_result
        new_motion_plan_keys.append(new_motion_plan_key)
        if motion_plan_result.task_success:
            any_success = True
            problems[problem_id].solved_motion_plan_results[new_motion_plan_key] = motion_plan_result

        print(f"  Motion plan result: task_success: {motion_plan_result.task_success}")
        print(f"  Total Actions Taken: {motion_plan_result.total_trajs_sampled}")
        print(f"  Total Time Taken: {end_time - start_time:.3f}s")

        if motion_plan_result.last_failed_operator:
            print(f"  Failed at operator: {motion_plan_result.last_failed_operator + 1} / {len(motion_plan_result.pddl_plan.plan)} operators in task plan.")
    return any_success, new_motion_plan_keys, used_mock


def mock_evaluate_motion_plans_and_costs_for_problems(output_filepath, output_directory, problems):
    unsolved_problems = set()
    with open(os.path.join(output_directory, output_filepath), "r") as f:
        output_json = json.load(f)
        print(f"Now in: mock_evaluate_motion_plans_and_costs_for_problems: from {os.path.join(output_directory, output_filepath)}")
    for plan in output_json:
        if plan["file_name"] in problems:
            problem = problems[plan["file_name"]]
            if len(plan["motion_plans"]) == 0:
                unsolved_problems.add(plan["file_name"])
            for plan_json in plan["motion_plans"]:
                # This updates the evaluated PDDL task plans that succeeded.
                motion_plan_result = MotionPlanResult.from_json(plan_json)
                problem.evaluated_motion_planner_results[(plan_json["goal"], plan_json["plan"])] = motion_plan_result
                if motion_plan_result.task_success:
                    any_success = True
                    problem.solved_motion_plan_results[
                        (plan_json["goal"], plan_json["plan"])  # The actual goal and task plan that we planned for.
                    ] = motion_plan_result

    print(f"After initialization, there are {len([p for p in problems if len(problems[p].evaluated_pddl_plans) > 0])} problems with plans.")
    return unsolved_problems


@functools.lru_cache()
def get_mocked_motion_plan_file(output_directory, plan_pass_identifier):
    filepath = os.path.join(output_directory, f"mocked_motion_plan_{plan_pass_identifier}.pkl")
    if not os.path.exists(filepath):
        return dict()
    with open(filepath, "rb") as f:
        return pickle.load(f)


def mock_motion_plan_for_problem_single(problem_id, pddl_goal, pddl_plan, output_directory, plan_pass_identifier):
    plans = get_mocked_motion_plan_file(output_directory, plan_pass_identifier)
    identifier = (problem_id, pddl_goal, pddl_plan.plan_string)
    if identifier in plans:
        return plans[identifier]
    return None


def checkpoint_motion_plan_for_problem_single(problem_id, pddl_goal, pddl_plan, output_directory, plan_pass_identifier, motion_plan_result):
    filepath = os.path.join(output_directory, f"mocked_motion_plan_{plan_pass_identifier}.pkl")

    plans = get_mocked_motion_plan_file(output_directory, plan_pass_identifier)
    identifier = (problem_id, pddl_goal, pddl_plan.plan_string)
    plans[identifier] = motion_plan_result
    with open(filepath, "wb") as f:
        pickle.dump(plans, f)
