import os
import alfred.alfredplanner as alfredplanner

from llm_operators.pddl import PDDLPlan
from llm_operators.experiment_utils import RANDOM_SEED
from llm_operators.motion_planner import MotionPlanResult


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


def evaluate_cw_motion_plans_and_costs_for_goal_plan(
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
            simulator.move_right()
        elif action_name == "move-left":
            simulator.move_left()
        elif action_name == "move-to":
            simulator.move_to(int(action_args[1][1:]))
        elif action_name == "pick-up":
            try:
                simulator.pick_up(
                    int(_find_string_start_with(action_args, "i", first=True)[1:]),
                    _find_string_start_with(action_args, "o", first=True),
                )
            except KeyError as e:
                print(f'  pick-up {action_args} failed. Reason: {e}')
                pass
        elif action_name == "place-down":
            simulator.place_down(
                int(_find_string_start_with(action_args, "i", first=True)[1:]),
            )
        elif action_name.startswith('mine'):
            # Trying mining.
            inventory_indices = [int(x[1:]) for x in _find_string_start_with(action_args, "i")]
            object_indices = _find_string_start_with(action_args, "o")

            hypothetical_object = [x for x in object_indices if x in simulator.hypothetical]
            if len(hypothetical_object) != 1:
                if verbose:
                    print("  Hypothetical object not found.", object_indices)
                last_failed_operator = i
                break
            hypothetical_object = hypothetical_object[0]

            empty_inventory = [x for x in inventory_indices if simulator.inventory[x] is None]
            if len(empty_inventory) != 1:
                if verbose:
                    print("  Empty inventory not found.", inventory_indices)
                last_failed_operator = i
                break
            empty_inventory = empty_inventory[0]

            target_object = [
                x for x in object_indices if x in simulator.objects and simulator.objects[x][1] == simulator.agent_pos
            ]
            if len(target_object) != 1:
                if verbose:
                    print("  Target object not found.", object_indices)
                last_failed_operator = i
                break
            target_object = target_object[0]

            tool_inventory = list(set(inventory_indices) - set([empty_inventory]))

            if verbose:
                print("  Mining", empty_inventory, hypothetical_object, target_object, tool_inventory)
            rv = simulator.mine(target_object, empty_inventory, hypothetical_object, tool_inventory=tool_inventory[0] if len(tool_inventory) > 0 else None)

            if not rv:
                last_failed_operator = i
        elif action_name.startswith('craft'):
            inventory_indices = [int(x[1:]) for x in _find_string_start_with(action_args, "i")]
            object_indices = _find_string_start_with(action_args, "o")

            hypothetical_object = [x for x in object_indices if x in simulator.hypothetical]
            if len(hypothetical_object) != 1:
                if verbose:
                    print("  Hypothetical object not found.", object_indices)
                last_failed_operator = i
                break
            hypothetical_object = hypothetical_object[0]

            empty_inventory = [x for x in inventory_indices if simulator.inventory[x] is None]
            if len(empty_inventory) != 1:
                if verbose:
                    print("  Empty inventory not found.", inventory_indices)
                last_failed_operator = i
                break
            empty_inventory = empty_inventory[0]

            target_object = [ x for x in object_indices if x in simulator.objects and simulator.objects[x][1] == simulator.agent_pos ]
            if len(target_object) != 1:
                if verbose:
                    print("  Target object not found.", object_indices)
                last_failed_operator = i
                break
            target_object = target_object[0]

            ingredients = list(set(inventory_indices) - set([empty_inventory]))

            # from llm_operators.experiment_utils import run_ipdb
            # run_ipdb()
            target_type = None
            hypothetical_object_index = action_args.index(hypothetical_object)
            hypothetical_object_varname = domain.operators[action_name].arguments[hypothetical_object_index].name
            for effect in domain.operators[action_name].effects:
                if (
                    effect.assign_expr.predicate.function.name == 'object-of-type' and
                    effect.assign_expr.predicate.arguments[0].name == hypothetical_object_varname and
                    effect.assign_expr.predicate.arguments[1].__class__.__name__ == 'ObjectConstantExpression' and
                    effect.assign_expr.value.__class__.__name__ == 'ConstantExpression' and
                    effect.assign_expr.value.constant.item() == 1
                ):
                    print('  Found target type', effect.assign_expr)
                    target_type = effect.assign_expr.predicate.arguments[1].name
                    break

            if verbose:
                print("  Crafting", empty_inventory, hypothetical_object, target_object, ingredients, f'target_type={target_type}')
            rv = simulator.craft(target_object, empty_inventory, hypothetical_object, ingredients_inventory=ingredients, target_type=target_type)

            if not rv:
                last_failed_operator = i
        else:
            last_failed_operator = i

    if last_failed_operator is not None:
        return MotionPlanResult(
            pddl_plan=pddl_plan,
            task_success=False,
            last_failed_operator=last_failed_operator,
            max_satisfied_predicates=None,
        )

    gt_pddl_problem = problems[problem_id].ground_truth_pddl_problem

    if '(exists' in gt_pddl_problem.ground_truth_goal:
        gt_goal = list()
        gt_goal_conjunction_part = gt_pddl_problem.ground_truth_goal.split('(and ')[1]
        gt_goal_conjunction_part = gt_goal_conjunction_part.strip()
        while gt_goal_conjunction_part[0] == '(':
            end_index = _find_first_matched_right_bracket(gt_goal_conjunction_part, 0)
            gt_goal.append(gt_goal_conjunction_part[1:end_index])
            gt_goal_conjunction_part = gt_goal_conjunction_part[end_index + 1:].strip()
        gt_goal_variables = [('?i', 'inventory'), ('?o', 'object')]
        satisfied = simulator.goal_satisfied_existential(gt_goal, gt_goal_variables)
    else:
        gt_goal = [x[1:-1] for x in gt_pddl_problem.ground_truth_goal_list]
        satisfied = simulator.goal_satisfied(gt_goal)

    return MotionPlanResult(
        pddl_plan=pddl_plan,
        task_success=satisfied,
    )


def _find_string_start_with(list_of_string, start, first=False):
    rv = list()
    for s in list_of_string:
        if s.startswith(start):
            if first:
                return s
            rv.append(s)
    return rv


def _find_first_matched_right_bracket(string, start_index):
    assert string[start_index] == '('
    count = 1
    for i in range(start_index + 1, len(string)):
        if string[i] == '(':
            count += 1
        elif string[i] == ')':
            count -= 1
            if count == 0:
                return i
    return None
