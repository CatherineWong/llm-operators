"""
task_planner.py 
Utilities for generating task level plans.
"""
from pddlgym_planners.fd import FD
from pddlgym_planners.planner import PlanningFailure, PlanningTimeout
from tempfile import NamedTemporaryFile
from pddl import PDDLPlan

TASK_PLANNER_FD = "task_planner_fd"


def evaluate_task_plans_and_costs_for_problems(
    pddl_domain, problems, command_args, verbose=False
):
    """
    Runs task planner to evaluate task plans for a set of planning problems, given a PDDL domain.
    :ret: problems updated with PDDL plans.
    """
    print(f"evaluate_task_plans_and_costs_for_problems on {len(problems)}.")
    if verbose:
        print(f"Use ground truth goals? {command_args.debug_ground_truth_goals}")
    for problem_id in problems:
        run_planner(
            pddl_domain=pddl_domain,
            problem=problems[problem_id],
            planner_type=TASK_PLANNER_FD,
            verbose=verbose,
            debug_ground_truth_goals=command_args.debug_ground_truth_goals,
        )


def run_planner(
    pddl_domain,
    problem,
    planner_type=TASK_PLANNER_FD,
    verbose=False,
    debug_ground_truth_goals=False,
):
    """
    pddl_domain: Domain object.
    problem: Problem object.
    planner_type: string indicating which planenr to use.

    :ret: Attempts to run planner on each goal in problem.proposed_pddl_goals.
    
    Updates problem.evaluated_pddl_plans to {
        goal : PDDLPlan
    } if a PDDLPlan is found, along with a score for this plan.
    """

    if debug_ground_truth_goals:
        goals = [problem.ground_truth_pddl_problem.ground_truth_goal]
    else:
        goals = problem.proposed_pddl_goals
    for goal in goals:
        current_problem_string = problem.ground_truth_pddl_problem.get_pddl_string_with_proposed_goal(
            proposed_goal=goal
        )
        if planner_type != TASK_PLANNER_FD:
            assert False

        # Get domain strings. Pick the first one that parses.
        current_domain_string = pddl_domain.to_string(
            ground_truth_operators=False,
            current_operators=True,
            proposed_operators=pddl_domain.proposed_operators.keys(),
        )
        success, raw_plan_list = fd_plan_from_strings(
            domain_str=current_domain_string, problem_str=current_problem_string
        )
        import pdb

        pdb.set_trace()
        # Convert the planner into a plan object.
        # TBD: include the exact sequence of operators.
        if success:
            plan_string = "\n".join(["(" + a + ")" for a in raw_plan_list])
            pddl_plan = PDDLPlan(plan_string=plan_string, pddl_domain=pddl_domain)
            problem.evaluated_pddl_plans[goal] = pddl_plan
    if verbose:
        print(
            f"Found {len(problem.evaluated_pddl_plans)}/{len(problem.proposed_pddl_goals)} evaluated plans for proposed goals"
        )


def fd_plan_from_strings(domain_str, problem_str):
    with NamedTemporaryFile(mode="w") as domain_file, NamedTemporaryFile(
        mode="w"
    ) as problem_file:
        domain_file.write(domain_str)
        problem_file.write(problem_str)
        domain_file.flush()
        problem_file.flush()
        success, out = fd_plan_from_file(domain_file.name, problem_file.name)
        return (success, out)


def fd_plan_from_file(domain_fname, problem_fname):
    # TBD: don't use PDDL gym planner, use original FD.
    fd_planner = FD(alias_flag='--alias "lama-first"')
    try:

        plan = fd_planner.plan_from_pddl(domain_fname, problem_fname, timeout=5)
    except PlanningFailure as pf:
        return False, pf
    except PlanningTimeout as pt:
        print("timed out")
        return False, pt
    return True, plan
