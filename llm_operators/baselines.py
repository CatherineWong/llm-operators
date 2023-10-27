import frozendict
from llm_operators import motion_planner
from llm_operators import pddl

import os.path as osp
import sys
import argparse
import llm_operators.codex as codex
import llm_operators.datasets as datasets
import llm_operators.experiment_utils as experiment_utils
from llm_operators.motion_planner import MotionPlanResult

import sys
if sys.version_info.minor < 9:
    print("Warning: using Python < 3.9 to run ALFRED: cannot run craftingworld typing.")
else:
    from llm_operators.datasets.crafting_world_skill_lib import *
    import llm_operators.datasets.crafting_world as crafting_world
    from llm_operators.datasets.crafting_world_gen.utils import pascal_to_underline
    from llm_operators.datasets.crafting_world import CraftingWorld20230204Simulator, local_search_for_subgoal, SimpleConjunction
    crafting_world.SKIP_CRAFTING_LOCATION_CHECK = True
    print('Skipping location check in Crafting World.')

import sys
if sys.version_info.minor < 9:
    print("Warning: using Python < 3.9 to run ALFRED: cannot run craftingworld typing.")
else:
    CONCEPTS_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "../concepts")
    print("Adding concepts path: {}".format(CONCEPTS_PATH))
    sys.path.insert(0, CONCEPTS_PATH)

def load_state_from_problem(pds_domain, problem_record, pddl_goal=None):
    pddl_problem = problem_record.ground_truth_pddl_problem
    current_problem_string = pddl_problem.get_ground_truth_pddl_string()
    problem = pds.load_problem_string(current_problem_string, pds_domain, return_tensor_state=False)

    gproblem = OnTheFlyGStripsProblem.from_domain_and_problem(pds_domain, problem)
    simulator = CraftingWorld20230204Simulator()
    simulator.reset_from_state(gproblem.objects, gproblem.initial_state)

    gt_goal = [x[1:-1] for x in pddl_problem.ground_truth_goal_list]

    return simulator, gt_goal

#### Baseline: Motion plan directly on the goal.
def _run_motion_plan_directly_on_goal(dataset_name, pddl_domain, problem_idx, problem_id, planning_problems, args, curr_iteration, output_directory, plan_pass_identifier, plan_attempt_idx, goal_idx, rng, split):
    problems = planning_problems[split]
    any_motion_planner_success = False
    if not "alfred" in dataset_name:
        # This was implemented separately for Minecraft
        assert False

    for idx, proposed_pddl_goal in enumerate(problems[problem_id].proposed_pddl_goals):
        print(f"Evaluating directly on [{idx+1}/{len(problems[problem_id].proposed_pddl_goals)}] LLM goals.")
        task_plans = {
            proposed_pddl_goal : proposed_pddl_goal
        }
        any_motion_planner_success, new_motion_plan_keys, used_motion_mock = motion_planner.attempt_motion_plan_for_problem(
                    pddl_domain=pddl_domain,
                    problem_idx=problem_idx,
                    problem_id=problem_id,
                    problems=planning_problems[split],
                    dataset_name=args.dataset_name,
                    new_task_plans=task_plans,
                    use_mock=args.debug_mock_motion_plans,
                    command_args=args,
                    curr_iteration=curr_iteration,
                    output_directory=output_directory,
                    plan_pass_identifier=plan_pass_identifier,
                    plan_attempt_idx=plan_attempt_idx,
                    resume=args.resume,
                    resume_from_iteration=args.resume_from_iteration,
                    resume_from_problem_idx=args.resume_from_problem_idx,
                    debug_skip=args.debug_skip_motion_plans,
                    verbose=args.verbose,
                    motion_plan_directly_on_goal=args.motion_plan_directly_on_goal # Baseline -- we skip task proposal if so.
                )

### Baseline: 
def _run_llm_propose_task_predicates_motion_planner(dataset_name, pddl_domain, problem_idx, problem_id, planning_problems, args, curr_iteration, output_directory, plan_pass_identifier, plan_attempt_idx, goal_idx, rng, split):
    problems = planning_problems[split]
    any_motion_planner_success = False
    if "alfred" in dataset_name:
        
        for idx, llm_task_plans in enumerate(problems[problem_id].proposed_pddl_task_predicates):
            print(f"Evaluating [{idx+1}/{len(problems[problem_id].proposed_pddl_task_predicates)}] LLM task plans.")
            # Sham task plans following the task planner format.
            task_plans = {
            problems[problem_id].ground_truth_pddl_problem.ground_truth_goal:  llm_task_plans
            }
            any_motion_planner_success, new_motion_plan_keys, used_motion_mock = motion_planner.attempt_motion_plan_for_problem(
                    pddl_domain=pddl_domain,
                    problem_idx=problem_idx,
                    problem_id=problem_id,
                    problems=planning_problems[split],
                    dataset_name=args.dataset_name,
                    new_task_plans=task_plans,
                    use_mock=args.debug_mock_motion_plans,
                    command_args=args,
                    curr_iteration=curr_iteration,
                    output_directory=output_directory,
                    plan_pass_identifier=plan_pass_identifier,
                    plan_attempt_idx=plan_attempt_idx,
                    resume=args.resume,
                    resume_from_iteration=args.resume_from_iteration,
                    resume_from_problem_idx=args.resume_from_problem_idx,
                    debug_skip=args.debug_skip_motion_plans,
                    verbose=args.verbose,
                    llm_propose_task_predicates=args.llm_propose_task_predicates # Baseline -- we skip task proposal if so.
                )
    else:
        current_domain_string = pddl_domain.to_string(
        ground_truth_operators=False,
        current_operators=True,
        )
        pds_domain = pds.load_domain_string(current_domain_string)

        problem = problems[problem_id]
        simulator, gt_goal = load_state_from_problem(pds_domain, problem)
        any_motion_planner_success = True
        for idx, subgoal_sequence in enumerate(problems[problem_id].proposed_pddl_task_predicates):
            print(f"Evaluating [{idx+1}/{len(problems[problem_id].proposed_pddl_task_predicates)}] LLM task plans.")
            
            print(f"Ground truth subgoal sequence is: {problem.ground_truth_subgoal_sequence}")
            print(f"LLM proposed subgoal sequence: {subgoal_sequence}")
            for subgoal in subgoal_sequence:
                print('  Subgoal: {}'.format(subgoal))
                rv = local_search_for_subgoal(simulator, SimpleConjunction(subgoal), max_steps=1e3)
                if rv is None:
                    print('    Failed to achieve subgoal: {}'.format(subgoal))
                    any_motion_planner_success = False
                    break
                simulator, _ = rv

            if any_motion_planner_success:
                any_motion_planner_success = simulator.goal_satisfied(gt_goal)
                print('Success: {}'.format(any_motion_planner_success))
                problems[problem_id].solved_motion_plan_results[(str(gt_goal), subgoal_sequence)]  = MotionPlanResult(
                                        pddl_plan=pddl.PDDLPlan(plan_string=""),
                                        task_success=any_motion_planner_success,
                                        last_failed_operator=-1,
                                        max_satisfied_predicates=None,
                                    )
            if any_motion_planner_success:
                return any_motion_planner_success

    return any_motion_planner_success


def _run_llm_propose_code_policies_motion_planner(dataset_name, pddl_domain, problem_idx, problem_id, planning_problems, args, curr_iteration, output_directory, plan_pass_identifier, plan_attempt_idx, goal_idx, rng, split):
    problems = planning_problems[split]
    any_motion_planner_success = False
    if "alfred" in dataset_name:
        new_motion_plan_keys = []
        for idx, llm_code_policies in enumerate(problems[problem_id].proposed_code_policies):
            print(f"Evaluating [{idx+1}/{len(problems[problem_id].proposed_code_policies)}] LLM proposed code policies.")
            # Sham task plans following the task planner format.
            code_policies = {
            problems[problem_id].ground_truth_pddl_problem.ground_truth_goal:  llm_code_policies
            }
            try:
                any_motion_planner_success, new_motion_plan_keys, used_motion_mock = motion_planner.attempt_motion_plan_for_problem(
                        pddl_domain=pddl_domain,
                        problem_idx=problem_idx,
                        problem_id=problem_id,
                        problems=planning_problems[split],
                        dataset_name=args.dataset_name,
                        new_task_plans=code_policies,
                        use_mock=args.debug_mock_motion_plans,
                        command_args=args,
                        curr_iteration=curr_iteration,
                        output_directory=output_directory,
                        plan_pass_identifier=plan_pass_identifier,
                        plan_attempt_idx=plan_attempt_idx,
                        resume=args.resume,
                        resume_from_iteration=args.resume_from_iteration,
                        resume_from_problem_idx=args.resume_from_problem_idx,
                        debug_skip=args.debug_skip_motion_plans,
                        verbose=args.verbose,
                        llm_propose_code_policies=args.llm_propose_code_policies # Baseline -- we skip task proposal if so.
                    )
            except:
                new_motion_plan_keys = []
                continue
    else:
        new_motion_plan_keys = []
        current_domain_string = pddl_domain.to_string(
        ground_truth_operators=False,
        current_operators=True,
        )
        pds_domain = pds.load_domain_string(current_domain_string)
        problem = problems[problem_id]
        
        _, gt_goal = load_state_from_problem(pds_domain, problem)

        for idx, policy_sequence in enumerate(problems[problem_id].proposed_code_policies):
            print(f"Evaluating [{idx+1}/{len(problems[problem_id].proposed_code_policies)}] LLM code policies.")
            motion_plan_result = evaluate_crafting_world_code_policy(policy_sequence, pds_domain, problem)

            # Boiler plate to update the problem so we can keep using the old score keeper.
            new_motion_plan_key = (str(gt_goal), motion_plan_result.pddl_plan.plan_string)
            problems[problem_id].evaluated_motion_planner_results[new_motion_plan_key] = motion_plan_result
            new_motion_plan_keys.append(new_motion_plan_key)
            if motion_plan_result.task_success:
                problems[problem_id].solved_motion_plan_results[new_motion_plan_key] = motion_plan_result
                break

    # Update the global operator scores from the problem.
    pddl.update_pddl_domain_and_problem(
        pddl_domain=pddl_domain,
        problem_idx=problem_idx,
        problem_id=problem_id,
        problems=planning_problems[split],
        new_motion_plan_keys=new_motion_plan_keys,
        command_args=args,
        verbose=args.verbose,
    )

    return any_motion_planner_success


def evaluate_crafting_world_code_policy(policy_sequence, pds_domain, problem):
    plan = pddl.PDDLPlan.from_code_policy(code_policy=policy_sequence)
    global env_state
    env_state, gt_goal = load_state_from_problem(pds_domain, problem)
    for policy_idx, curr_policy in enumerate(policy_sequence):
        print(f"Now on policy: {policy_idx}/{len(policy_sequence)}: {curr_policy['action']}")
        try:
            curr_policy_success = attempt_execute_policy(policy=curr_policy)
        except:
            curr_policy_success = False
        if curr_policy_success == False:
            return MotionPlanResult(
                            pddl_plan=plan,
                            task_success=curr_policy_success,
                            last_failed_operator=policy_idx,
                            max_satisfied_predicates=None,
            )
    # Check if goal is satisfied.
    curr_policy_success = env_state.goal_satisfied(gt_goal)
    return MotionPlanResult(
            pddl_plan=plan,
            task_success=curr_policy_success,
            last_failed_operator=None,
            max_satisfied_predicates=None,
    )
    

def attempt_execute_policy(policy):
    success = False
    try:
        # Create the argument map.
        argument_bindings = list(zip(policy['argument_names'], policy['ground_arguments']))
        argument_string = ", ".join([f"{a_name}={a_value}" for (a_name, a_value) in argument_bindings])

        # Note that this does assume that each function call itself does not depend on previously defined functions. We should check if this is true for our purposes.
        class ExecutionResult():
            def __init__(self):
                python_function = f"""def temp_func({argument_string}): \n{policy['body']}\nself.success = temp_func()"""
                exec(python_function)
        ex_result = ExecutionResult()
        print(f"Execution result: { ex_result.success}")
        return ex_result.success
    except:
        print("Failed to execute the function defined in code.")
        return success

