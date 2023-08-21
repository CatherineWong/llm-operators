"""
motion_planner.py
Utilities for generating motion plans.
"""

import json
import os
import alfred.alfredplanner as alfredplanner

from llm_operators.pddl import PDDLPlan
from llm_operators.experiment_utils import RANDOM_SEED


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
    command_args,
    verbose=False,
    output_directory=None,
    use_mock=False,
    debug_skip=False,
    plan_attempt_idx=0,
    dataset_name="",
    new_task_plans=None,
):
    """Attempts to motion plan for a single problem. This attempts the planner on any proposed goals, and any proposed task plans for those goals."""
    if plan_attempt_idx == 0:
        print(
            f"motion_planner.attempt_motion_plan_for_problem: attempt {plan_attempt_idx} : {problem_idx} / {len(problems)}"
        )
    else:
        print(f"\tmotion_planner.attempt_motion_plan_for_problem: attempt {plan_attempt_idx}")
    experiment_tag = "" if len(command_args.experiment_name) < 1 else f"{command_args.experiment_name}_"

    output_filepath = f"{experiment_tag}motion_plans.json"
    if use_mock:
        try:
            unsolved_problems = mock_evaluate_motion_plans_and_costs_for_problems(
                output_filepath, output_directory, problems
            )
            if problem_id in unsolved_problems or len(problems[problem_id].evaluated_motion_planner_results) > 0:
                print("Mock found for motion plan, continuing...")
                return
            else:
                print("Mock not found for motion plan, continuing...")
        except:
            print("Mock not found for motion plan, continuing...")

    any_success = False
    new_motion_plan_keys = []
    for pddl_goal, pddl_plan in new_task_plans.items():
        if "alfred" in dataset_name:
            motion_plan_result = evaluate_alfred_motion_plans_and_costs_for_goal_plan(
                problem_id,
                problems,
                pddl_goal,
                pddl_plan,
                pddl_domain,
                verbose,
                debug_skip=debug_skip,
                motionplan_search_type=command_args.motionplan_search_type,
            )
        elif dataset_name == "crafting_world_20230204_minining_only":
            motion_plan_result = evaluate_cw_20230204_motion_plans_and_costs_for_goal_plan(
                problem_id,
                problems,
                pddl_goal,
                pddl_plan,
                pddl_domain,
                verbose,
                debug_skip=debug_skip,
            )
        new_motion_plan_key = (pddl_goal, motion_plan_result.pddl_plan.plan_string)
        problems[problem_id].evaluated_motion_planner_results[
            new_motion_plan_key  # The actual goal and task plan that we planned for.
        ] = motion_plan_result
        new_motion_plan_keys.append(new_motion_plan_key)
        if motion_plan_result.task_success:
            any_success = True

        print("=============================================")
        print(f"Problem Number: {problem_idx}")
        print(f"Problem ID: {problem_id}")
        print(f"Motion plan result: task_success: {motion_plan_result.task_success}")
        print(f"Total Actions Taken: {motion_plan_result.total_trajs_sampled}")

        if motion_plan_result.last_failed_operator:
            print(
                f"Failed at operator: {motion_plan_result.last_failed_operator + 1} / {len(motion_plan_result.pddl_plan.plan)} operators in task plan."
            )
        print("=============================================")
    return any_success, new_motion_plan_keys


def evaluate_motion_plans_and_costs_for_problems(
    curr_iteration,
    pddl_domain,
    problems,
    command_args,
    verbose=False,
    output_directory=None,
    use_mock=False,
    debug_skip=False,
    dataset_name="",
):
    """
    Runs a motion planner.
    """
    if "alfred" in dataset_name:
        evaluate_alfred_motion_plans_and_costs_for_problems(
            curr_iteration,
            pddl_domain,
            problems,
            command_args,
            verbose=verbose,
            output_directory=output_directory,
            use_mock=use_mock,
            debug_skip=debug_skip,
        )
    elif dataset_name == "crafting_world_20230204_minining_only":
        evaluate_cw_20230204_motion_plans_and_costs_for_problems(
            curr_iteration,
            pddl_domain,
            problems,
            command_args,
            verbose=verbose,
            output_directory=output_directory,
            use_mock=use_mock,
            debug_skip=debug_skip,
        )
    else:
        print(f"Unsupported dataset name: {dataset_name}")
        assert False


def mock_evaluate_motion_plans_and_costs_for_problems(output_filepath, output_directory, problems):
    unsolved_problems = set()
    with open(os.path.join(output_directory, output_filepath), "r") as f:
        output_json = json.load(f)
        print(
            f"Now in: mock_evaluate_motion_plans_and_costs_for_problems: from {os.path.join(output_directory, output_filepath)}"
        )
    for plan in output_json:
        if plan["file_name"] in problems:
            problem = problems[plan["file_name"]]
            if len(plan["motion_plans"]) == 0:
                unsolved_problems.add(plan["file_name"])

            for plan_json in plan["motion_plans"]:
                # This updates the evaluated PDDL task plans that succeeded.
                problem.evaluated_motion_planner_results[
                    (plan_json["goal"], plan_json["plan"])
                ] = MotionPlanResult.from_json(plan_json)

    print(
        f"After initialization, there are {len([p for p in problems if len(problems[p].evaluated_pddl_plans) > 0])} problems with plans."
    )
    return unsolved_problems


def evaluate_alfred_motion_plans_and_costs_for_problems(
    curr_iteration,
    pddl_domain,
    problems,
    command_args,
    verbose=False,
    output_directory=None,
    use_mock=False,
    debug_skip=False,
):
    print(f"evaluate_motion_plans_and_costs_for_problems on {len(problems)} problems.")
    experiment_tag = "" if len(command_args.experiment_name) < 1 else f"{command_args.experiment_name}_"
    output_filepath = f"{experiment_tag}motion_plans.json"

    for max_problems, problem_id in enumerate(problems):
        for pddl_goal in problems[problem_id].evaluated_pddl_plans:
            for pddl_plan in problems[problem_id].evaluated_pddl_plans[pddl_goal]:
                # run_motion_planner
                # not implemented envname to task_id
                # recording videos. quicktime player.
                motion_plan_result = evaluate_alfred_motion_plans_and_costs_for_goal_plan(
                    problem_id,
                    problems,
                    pddl_goal,
                    pddl_plan,
                    pddl_domain,
                    verbose,
                    debug_skip=debug_skip,
                )
                problems[problem_id].evaluated_motion_planner_results[pddl_goal] = motion_plan_result
                if motion_plan_result.task_success:
                    problems[problem_id].best_evaluated_plan_at_iteration = curr_iteration
                if verbose:
                    print(f"Motion plan result: task_success: {motion_plan_result.task_success}")
                    if motion_plan_result.last_failed_operator:
                        print(
                            f"Successfully executed: {motion_plan_result.last_failed_operator + 1} / {len(motion_plan_result.pddl_plan)} operators in final task sequence.\n\n\n"
                        )


def evaluate_alfred_motion_plans_and_costs_for_goal_plan(
    problem_id,
    problems,
    pddl_goal,
    pddl_plan,
    pddl_domain,
    verbose,
    debug_skip=False,
    motionplan_search_type="bfs",
):
    if verbose:
        print(f"Motion planning for: {problem_id}")
        print(f"Proposed goal is: ")
        print(pddl_goal)
        print(f"Ground truth oracle goal is: ")
        print(problems[problem_id].ground_truth_pddl_problem.ground_truth_goal)

    # Convert plan to sequential plan predicates. Returns a pruned PDDL plan that does not include operators we didn't execute.
    task_plan_json, pruned_pddl_plan = pddl_plan.to_task_plan_json(
        problem=problems[problem_id],
        pddl_domain=pddl_domain,
        remove_alfred_object_ids=True,
        remove_alfred_agent=True,
    )
    operator_sequence = task_plan_json["operator_sequence"]
    # This is the ground truth goal according to ALFRED.
    goal_ground_truth_predicates = task_plan_json["goal_ground_truth_predicates"]
    if debug_skip:
        return MotionPlanResult(
            pddl_plan=pruned_pddl_plan,
            task_success=True,
            last_failed_operator=None,
            max_satisfied_predicates=operator_sequence[-1][PDDLPlan.PDDL_POSTCOND_GROUND_PREDICATES][-1],
        )
    else:
        # Run the motion planner.
        dataset_split = os.path.split(problem_id)[0]
        task_name = os.path.join(*os.path.split(problem_id)[1:])
        if verbose:
            print("Attempting to execute the following motion plan:")
            for pred in operator_sequence:
                print(f"{pred}\n")

            print("Ground truth PDDL plan is: ")
            print(problems[problem_id].ground_truth_pddl_plan.plan_string)

            print("Goal ground truth predicates that will be evaluated: ")
            for pred in goal_ground_truth_predicates:
                print(f"{pred}\n")
        alfred_motion_task = {
            "task": task_name,
            "repeat_idx": 0,  # How do we know which one it is?
        }
        raw_motion_plan_result = alfredplanner.run_motion_planner(
            task=alfred_motion_task,
            operator_sequence=operator_sequence,
            goal_ground_predicates=goal_ground_truth_predicates,
            robot_init=RANDOM_SEED,
            dataset_split=dataset_split,
            verbose=verbose,
            motionplan_search_type=motionplan_search_type,
        )
        return MotionPlanResult(
            pddl_plan=pruned_pddl_plan,
            task_success=raw_motion_plan_result["task_success"],
            last_failed_operator=raw_motion_plan_result["last_failed_operator"],
            max_satisfied_predicates=raw_motion_plan_result["max_satisfied_predicates"],
            total_trajs_sampled=raw_motion_plan_result["total_trajs_sampled"],
        )


def evaluate_cw_20230204_motion_plans_and_costs_for_problems(
    curr_iteration,
    pddl_domain,
    problems,
    command_args,
    verbose=False,
    output_directory=None,
    use_mock=False,
    debug_skip=False,
):
    print(f"evaluate_motion_plans_and_costs_for_problems on {len(problems)} problems.")
    experiment_tag = "" if len(command_args.experiment_name) < 1 else f"{command_args.experiment_name}_"
    output_filepath = f"{experiment_tag}motion_plans.json"

    if use_mock:
        # Not implemented
        assert False

    for max_problems, problem_id in enumerate(problems):
        for pddl_goal in problems[problem_id].evaluated_pddl_plans:
            pddl_plan = problems[problem_id].evaluated_pddl_plans[pddl_goal]
            if pddl_plan is not None and pddl_plan != {} and pddl_plan.plan is not None:
                motion_plan_result = evaluate_cw_20230204_motion_plans_and_costs_for_goal_plan(
                    # current_domain_string, current_problem_string, pddl_goal, pddl_plan
                    problem_id,
                    problems,
                    pddl_goal,
                    pddl_plan,
                    pddl_domain,
                    verbose,
                    debug_skip=debug_skip,
                )
                problems[problem_id].evaluated_motion_planner_results[pddl_goal] = motion_plan_result
                if motion_plan_result.task_success:
                    problems[problem_id].best_evaluated_plan_at_iteration = curr_iteration
                if verbose:
                    print(f"Motion plan result: task_success: {motion_plan_result.task_success}")
                    print(
                        f"Successfully executed: {motion_plan_result.last_failed_operator} / {len(motion_plan_result.pddl_plan.plan)} operators in task plan.\n\n\n"
                    )


def evaluate_cw_20230204_motion_plans_and_costs_for_goal_plan(
    problem_id,
    problems,
    pddl_goal,
    pddl_plan,
    pddl_domain,
    verbose,
    debug_skip=False,
):
    problem = problems[problem_id].ground_truth_pddl_problem
    current_problem_string = problem.get_pddl_string_with_proposed_goal(proposed_goal=pddl_goal)
    current_domain_string = pddl_domain.to_string(
        ground_truth_operators=False,
        current_operators=True,
        proposed_operators=pddl_domain.proposed_operators.keys(),
    )

    import concepts.pdsketch as pds

    domain = pds.load_domain_string(current_domain_string)
    problem = pds.load_problem_string(current_problem_string, domain, return_tensor_state=False)

    from concepts.pdsketch.strips.strips_grounding_onthefly import (
        OnTheFlyGStripsProblem,
        ogstrips_bind_arguments,
    )

    gproblem = OnTheFlyGStripsProblem.from_domain_and_problem(domain, problem)

    from llm_operators.datasets.crafting_world import CraftingWorld20230204Simulator

    simulator = CraftingWorld20230204Simulator()
    simulator.reset_from_state(gproblem.objects, gproblem.initial_state)

    last_failed_operator = None

    for i, action in enumerate(pddl_plan.plan):
        action_name = action[PDDLPlan.PDDL_ACTION]
        action_args = action[PDDLPlan.PDDL_ARGUMENTS]

        if action_name == "move-right":
            if verbose:
                print("move-right")
            simulator.move_right()
        elif action_name == "move-left":
            if verbose:
                print("move-left")
            simulator.move_left()
        elif action_name == "move-to":
            if verbose:
                print("move-to")
            simulator.move_to(int(action_args[1][1:]))
        elif action_name == "pick-up":
            if verbose:
                print("pick-up")
            simulator.pick_up(
                int(_find_string_start_with(action_args, "i", first=True)[1:]),
                _find_string_start_with(action_args, "o", first=True),
            )
        elif action_name == "place-down":
            if verbose:
                print("place-down")
            simulator.place_down(
                int(_find_string_start_with(action_args, "i", first=True)[1:]),
            )
        else:
            # Trying mining.
            inventory_indices = [int(x[1:]) for x in _find_string_start_with(action_args, "i")]
            object_indices = _find_string_start_with(action_args, "o")

            hypothetical_object = [x for x in object_indices if x in simulator.hypothetical]
            if len(hypothetical_object) != 1:
                if verbose:
                    print("Hypothetical object not found.", object_indices)
                last_failed_operator = i
                break
            hypothetical_object = hypothetical_object[0]

            empty_inventory = [x for x in inventory_indices if simulator.inventory[x] is None]
            if len(empty_inventory) != 1:
                if verbose:
                    print("Empty inventory not found.", inventory_indices)
                last_failed_operator = i
                break
            empty_inventory = empty_inventory[0]

            target_object = [
                x for x in object_indices if x in simulator.objects and simulator.objects[x][1] == simulator.agent_pos
            ]
            if len(target_object) != 1:
                if verbose:
                    print("Target object not found.", object_indices)
                last_failed_operator = i
                break
            target_object = target_object[0]

            tool_inventory = list(set(inventory_indices) - set([empty_inventory]))

            if verbose:
                print(
                    "Mining",
                    empty_inventory,
                    hypothetical_object,
                    target_object,
                    tool_inventory,
                )
            simulator.mine(
                target_object,
                empty_inventory,
                hypothetical_object,
                tool_inventory=tool_inventory[0] if len(tool_inventory) > 0 else None,
            )

    if last_failed_operator is not None:
        return MotionPlanResult(
            pddl_plan=pddl_plan,
            task_success=False,
            last_failed_operator=last_failed_operator,
            max_satisfied_predicates=None,
        )

    gt_pddl_problem = problems[problem_id].ground_truth_pddl_problem
    gt_goal = [x[1:-1] for x in gt_pddl_problem.ground_truth_goal_list]

    return MotionPlanResult(
        pddl_plan=pddl_plan,
        task_success=simulator.goal_satisfied(gt_goal),
    )


def _find_string_start_with(list_of_string, start, first=False):
    rv = list()
    for s in list_of_string:
        if s.startswith(start):
            if first:
                return s
            rv.append(s)
    return rv
