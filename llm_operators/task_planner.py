"""
task_planner.py
Utilities for generating task level plans.
"""

from collections import defaultdict
import os
import json
import random
from tempfile import NamedTemporaryFile
from typing import Optional, Sequence
import copy

from pddlgym_planners.fd import FD
from pddlgym_planners.planner import PlanningFailure, PlanningTimeout

from llm_operators.pddl import PDDLPlan

TASK_PLANNER_FD = "task_planner_fd"
TASK_PLANNER_PDSKETCH_ONTHEFLY = "task_planner_pdsketch_onthefly"


def attempt_task_plan_for_problem(
    pddl_domain,
    problem_idx,
    problem_id,
    problems,
    command_args,
    verbose=False,
    output_directory=None,
    use_mock=False,
    debug_skip=False,
    proposed_operators: Optional[Sequence[str]] = None,
    task_plan_with_constants=False,
    plan_attempt_idx=0,
    max_task_samples=4,
):
    """
    Evaluates planner to evaluate task plans for a single planning problems, given a PDDL domain.
    :ret: problem updated with PDDL plans.
    """
    if plan_attempt_idx == 0:
        print(
            f"task_planner.attempt_task_plan_for_problem: attempt {plan_attempt_idx} : {problem_idx} / {len(problems)}"
        )
    else:
        print(
            f"\ttask_planner.attempt_task_plan_for_problem: attempt {plan_attempt_idx}"
        )
    if debug_skip:
        print("\t...debug_skip.")

    experiment_tag = (
        ""
        if len(command_args.experiment_name) < 1
        else f"{command_args.experiment_name}_"
    )

    output_filepath = f"{experiment_tag}task_plans.json"

    if use_mock:
        try:
            mock_evaluate_task_plans_and_costs_for_problems(
                output_filepath, output_directory, problems
            )
            if len(problems[problem_id].evaluated_pddl_plans) > 0:
                return
            else:
                print("Mock not found for task plan, continuing...")
        except:
            print("Mock not found for task plan, continuing...")

    sample_operator_percent = 1.0 if plan_attempt_idx == 0 else 0.5
    # Don't get any proposed operators with negative scores.
    if proposed_operators is None:
        proposed_operators = set()
        for operator_name in pddl_domain.proposed_operators:
            for operator_body in pddl_domain.proposed_operators[operator_name]:
                if pddl_domain.operators_to_scores[(operator_name, operator_body)] >= 0:
                    proposed_operators.add(operator_name)

    print(f"\tsample_operator_percent: {sample_operator_percent}")
    any_success, new_evaluated_plans, problem_json = sample_task_plans_for_problem(
        pddl_domain=pddl_domain,
        problem=problems[problem_id],
        planner_type=command_args.planner,
        verbose=verbose,
        debug_ground_truth_goals=command_args.debug_ground_truth_goals,
        proposed_operators=proposed_operators,
        sample_operator_percent=sample_operator_percent,
    )
    if any_success:
        problems[problem_id].update_evaluated_pddl_plans(new_evaluated_plans)


def evaluate_task_plans_and_costs_for_problems(
    pddl_domain,
    problems,
    command_args,
    verbose=False,
    output_directory=None,
    use_mock=False,
    debug_skip=False,
    proposed_operators: Optional[Sequence[str]] = None,
    task_plan_with_constants=False,
    max_task_samples=4,
):
    """
    Batch evaluates planner to evaluate task plans for a set of planning problems, given a PDDL domain.

    For now, this just runs using the first operator definition.
    :ret: problems updated with PDDL plans.
    """
    if debug_skip:
        print(f"debug_skip_task_plans_and_costs_for_problems on {len(problems)}.")
        return
    print(f"evaluate_task_plans_and_costs_for_problems on {len(problems)}.")

    output_json = []
    experiment_tag = (
        ""
        if len(command_args.experiment_name) < 1
        else f"{command_args.experiment_name}_"
    )

    output_filepath = f"{experiment_tag}task_plans.json"

    if use_mock:
        mock_evaluate_task_plans_and_costs_for_problems(
            output_filepath, output_directory, problems
        )
        return

    if verbose:
        print(f"Use ground truth goals? {command_args.debug_ground_truth_goals}")

    total_solved_problems = 0
    for max_problems, problem_id in enumerate(problems):

        if verbose:
            print(
                f"\nNow on problem {max_problems} / {len(problems)}. Total solved problems so far: {total_solved_problems}"
            )
        any_success, new_evaluated_plans, problem_json = sample_task_plans_for_problem(
            pddl_domain=pddl_domain,
            problem=problems[problem_id],
            planner_type=command_args.planner,
            verbose=verbose,
            debug_ground_truth_goals=command_args.debug_ground_truth_goals,
            proposed_operators=proposed_operators,
            max_task_samples=max_task_samples,
        )
        problems[problem_id].update_evaluated_pddl_plans(new_evaluated_plans)
        if any_success:
            total_solved_problems += 1
        output_json.append(problem_json)
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)


def generate_random_proposed_operator_samples(
    proposed_operators, num_samples=4, sample_percent=0.5
):
    # Sample some percentage of the operator
    num_to_sample = int(sample_percent * len(proposed_operators))
    return [
        random.sample(proposed_operators, num_to_sample) for _ in range(num_samples)
    ]


def generate_random_proposed_operator_sample(proposed_operators, sample_percent=0.5):
    if sample_percent == 1.0:
        return proposed_operators
    # Sample some percentage of the operator
    num_to_sample = int(sample_percent * len(proposed_operators))
    return random.sample(proposed_operators, num_to_sample)


def sample_task_plans_for_problem(
    pddl_domain,
    problem,
    planner_type=TASK_PLANNER_FD,
    verbose=False,
    debug_ground_truth_goals=False,
    proposed_operators: Optional[Sequence[str]] = None,
    sample_operator_percent=1.0,
):
    """
    Uses a task_planner to propose samples, so we attempt planning using random subsets of 
    proposed_operator set to get a diverse set of plans.

    :ret: 
    any_success - whether any of the task plans succeeded.
    all_evaluated_plans: dict(goal : set(plans for this goal))
    overall_problem_json: serializable JSON format.
    """
    if proposed_operators is None:
        proposed_operators = pddl_domain.proposed_operators.keys()

    overall_problem_json = {"file_name": problem.problem_id, "plans": []}
    any_success = False
    all_evaluated_plans = defaultdict(set)

    # Sample a set of operators to try. Don't sample any operators with negative scores.
    sampled_proposed_operators = generate_random_proposed_operator_sample(
        proposed_operators, sample_percent=sample_operator_percent
    )
    success, evaluated_plans, _ = run_planner(
        pddl_domain=pddl_domain,
        problem=problem,
        planner_type=planner_type,
        verbose=verbose,
        debug_ground_truth_goals=debug_ground_truth_goals,
        proposed_operators=sampled_proposed_operators,
    )
    any_success = any_success or success
    for g in evaluated_plans:
        all_evaluated_plans[g].add(evaluated_plans[g])

    for g in all_evaluated_plans:
        for pddl_plan in all_evaluated_plans[g]:
            overall_problem_json["plans"].append({"goal": g, "plan": pddl_plan.plan})
        print(f"Found a total of {len(all_evaluated_plans[g])} unique plans for goal.")
    return any_success, all_evaluated_plans, overall_problem_json


def mock_evaluate_task_plans_and_costs_for_problems(
    output_filepath, output_directory, problems
):
    with open(os.path.join(output_directory, output_filepath), "r") as f:
        output_json = json.load(f)
        print(
            f"Now in: mock_evaluate_task_plans_and_costs_for_problems: from {os.path.join(output_directory, output_filepath)}"
        )
    for plan in output_json:
        if plan["file_name"] in problems:
            problem = problems[plan["file_name"]]
            for plan_json in plan["plans"]:
                # This updates the evaluated PDDL task plans that succeeded.
                problem.evaluated_pddl_plans[plan_json["goal"]].add(
                    PDDLPlan(plan=plan_json["plan"])
                )
    print(
        f"After initialization, there are {len([p for p in problems if len(problems[p].evaluated_pddl_plans) > 0])} problems with plans."
    )


def run_planner(
    pddl_domain,
    problem,
    planner_type=TASK_PLANNER_FD,
    verbose=False,
    debug_ground_truth_goals=False,
    proposed_operators: Optional[Sequence[str]] = None,
):
    """
    pddl_domain: Domain object.
    problem: Problem object.
    planner_type: string indicating which planenr to use.

    :ret: Attempts to run planner on each goal in problem.proposed_pddl_goals.
    any_success: whether any of the goals succeeded.
    evaluated_plans : {goal : Plan}
    """
    output_json = {"file_name": problem.problem_id, "plans": []}

    # Get domain strings. Pick the first one that parses
    current_domain_string = pddl_domain.to_string(
        ground_truth_operators=False,
        current_operators=True,
        proposed_operators=proposed_operators,
        show_constants=(not problem.constants_in_problem_file),
    )

    if debug_ground_truth_goals:
        goals = [problem.ground_truth_pddl_problem.ground_truth_goal]
    else:
        goals = problem.proposed_pddl_goals
    if len(goals) < 1:
        print("\t...no goals, skipping.")

    any_success = False
    evaluated_plans = dict()

    for goal in goals:
        if verbose:
            print(
                f"\tRunning planner with existing operators + {len(proposed_operators)} proposed operators: "
            )
            print(f"\t {pddl_domain.operators.keys()}")
            print(f"\t {proposed_operators}")
        current_problem_string = problem.ground_truth_pddl_problem.get_pddl_string_with_proposed_goal(
            proposed_goal=goal
        )
        if verbose:
            print("\t Ground truth goal: ")
            print("\t" + problem.ground_truth_pddl_problem.ground_truth_goal)
            print("\t Proposed goal:")
            print("\t" + goal)
        if planner_type == TASK_PLANNER_FD:
            success, plan_string = fd_plan_from_strings(
                domain_str=current_domain_string,
                problem_str=current_problem_string,
                verbose=verbose,
            )
        elif planner_type == TASK_PLANNER_PDSKETCH_ONTHEFLY:
            success, plan_string = pdsketch_onthefly_plan_from_strings(
                domain_str=current_domain_string, problem_str=current_problem_string
            )
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
        # Convert the planner into a plan object.
        if verbose:
            print(f"\tPlan success: {success}")
            print(f"\t Plan string: {plan_string}")
        if success:
            pddl_plan = PDDLPlan(plan_string=plan_string, pddl_domain=pddl_domain)
            evaluated_plans[goal] = pddl_plan
            output_json["plans"].append({"goal": goal, "plan": pddl_plan.plan})
            any_success = True
    return any_success, evaluated_plans, output_json


def fd_plan_from_strings(domain_str, problem_str, timeout=10, verbose=False):
    with NamedTemporaryFile(mode="w") as domain_file, NamedTemporaryFile(
        mode="w"
    ) as problem_file:
        domain_file.write(domain_str)
        problem_file.write(problem_str)
        domain_file.flush()
        problem_file.flush()
        success, out = fd_plan_from_file(
            domain_file.name, problem_file.name, timeout=timeout
        )
        return (success, out)


def fd_plan_from_file(domain_fname, problem_fname, timeout=5):
    # TBD: don't use PDDL gym planner, use original FD.
    fd_planner = FD(alias_flag='--alias "lama-first"')
    try:
        plan = fd_planner.plan_from_pddl(domain_fname, problem_fname, timeout=timeout)
        plan_string = "\n".join(["(" + a + ")" for a in plan])
    except PlanningFailure as pf:
        return False, pf
    except PlanningTimeout as pt:
        print("Time out")
        return False, pt
    return True, plan_string


def pdsketch_onthefly_plan_from_strings(domain_str, problem_str, timeout=10):
    import concepts.pdsketch as pds

    domain = pds.load_domain_string(domain_str)
    problem = pds.load_problem_string(problem_str, domain, return_tensor_state=False)

    from concepts.pdsketch.strips.strips_grounding_onthefly import (
        OnTheFlyGStripsProblem,
    )

    gproblem = OnTheFlyGStripsProblem.from_domain_and_problem(domain, problem)
    from concepts.pdsketch.strips.strips_grounding_onthefly import ogstrips_search

    plan = ogstrips_search(gproblem, timeout=timeout)

    if plan is None:
        return False, None
    return (
        True,
        "\n".join([op.to_applier_pddl_str(arguments) for op, arguments in plan]),
    )

