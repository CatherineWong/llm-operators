import frozendict
from llm_operators import motion_planner
from llm_operators import pddl

import os.path as osp
import sys
import argparse
import llm_operators.codex as codex
import llm_operators.datasets as datasets
import llm_operators.experiment_utils as experiment_utils
import llm_operators.datasets.crafting_world as crafting_world
from llm_operators.datasets.crafting_world_gen.utils import pascal_to_underline
from llm_operators.datasets.crafting_world import CraftingWorld20230204Simulator, local_search_for_subgoal, SimpleConjunction
from llm_operators.motion_planner import MotionPlanResult
from llm_operators.datasets.crafting_world_skill_lib import *

import concepts.pdsketch as pds
from concepts.pdsketch.strips.strips_grounding_onthefly import OnTheFlyGStripsProblem, ogstrips_bind_arguments

crafting_world.SKIP_CRAFTING_LOCATION_CHECK = True
print('Skipping location check in Crafting World.')

def load_state_from_problem(pds_domain, problem_record, pddl_goal=None):
    pddl_problem = problem_record.ground_truth_pddl_problem
    current_problem_string = pddl_problem.get_ground_truth_pddl_string()
    problem = pds.load_problem_string(current_problem_string, pds_domain, return_tensor_state=False)

    gproblem = OnTheFlyGStripsProblem.from_domain_and_problem(pds_domain, problem)
    simulator = CraftingWorld20230204Simulator()
    simulator.reset_from_state(gproblem.objects, gproblem.initial_state)

    gt_goal = [x[1:-1] for x in pddl_problem.ground_truth_goal_list]

    return simulator, gt_goal

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
        for idx, llm_code_policies in enumerate(problems[problem_id].proposed_code_policies):
            print(f"Evaluating [{idx+1}/{len(problems[problem_id].proposed_code_policies)}] LLM proposed code policies.")
            # Sham task plans following the task planner format.
            code_policies = {
            problems[problem_id].ground_truth_pddl_problem.ground_truth_goal:  llm_code_policies
            }
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
            motion_plan_result = evaluate_crafting_world_code_policy(policy_sequence, pds_domain, problem, pddl_domain)

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


def evaluate_crafting_world_code_policy(policy_sequence, pds_domain, problem, pddl_domain):
    plan = pddl.PDDLPlan.from_code_policy(code_policy=policy_sequence)
    global env_state
    env_state, gt_goal = load_state_from_problem(pds_domain, problem)

    # Action defintions
    action_definitions = []
    for policy_idx, curr_policy in enumerate(policy_sequence):
        print(f"Now on policy: {policy_idx}/{len(policy_sequence)}: {curr_policy['action']}")
        try:
            curr_policy_success = attempt_execute_policy(policy=curr_policy, action_definitions=action_definitions, pddl_domain=pddl_domain)

            if curr_policy[pddl.PDDLPlan.PDDL_ACTION_TYPE] == pddl.PDDLPlan.PDDL_ACTION_DEFINITION and curr_policy_success:
                print(f"Successful action definition, adding {curr_policy['action']} to action definitions")
                action_definitions.append(curr_policy)
        except:
            print("Policy unsuccessful.")
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
    print(f"Now evaluating goal: {curr_policy_success}.")
    return MotionPlanResult(
            pddl_plan=plan,
            task_success=curr_policy_success,
            last_failed_operator=None,
            max_satisfied_predicates=None,
    )

def create_function_definition(policy, pddl_domain):
    canonical_name = pddl_domain.code_skill_canonical_name_map.get(policy['action'], policy['action'])
    argument_string = ", ".join([f"{a_name}" for a_name in policy['argument_names']])
    python_function = f"""def {canonical_name}({argument_string}): \n{policy['body']}\n"""
    return python_function, canonical_name

def create_new_function_definition_test_string(action_definitions, new_function, pddl_domain):
    # Define all of the previous functions.
    existing_functions = [create_function_definition(action_definition, pddl_domain)[0] for action_definition in action_definitions]
    existing_functions_string = '\n'.join(existing_functions)
    new_function_definition, new_function_name = create_function_definition(new_function, pddl_domain)
    test_string = f"""{existing_functions_string}\n{new_function_definition}\nself.success = {new_function_name} is not None"""
    return test_string

def create_function_call_test_string(action_definitions, policy, pddl_domain):
    canonical_name = pddl_domain.code_skill_canonical_name_map.get(policy['action'], policy['action'])
    argument_bindings = list(zip(policy['argument_names'], policy['ground_arguments']))
    argument_string = ", ".join([f"{a_name}={a_value}" for (a_name, a_value) in argument_bindings])
    # Does it have a body?
    if len(policy['body']) > 0:
        function_call_string = f"""\ndef {canonical_name}({argument_string}): \n{policy['body']}\nself.success = {canonical_name}()"""
    else:
        function_call_string = f"""\nself.success = {canonical_name}({argument_string})"""
    
    # Create test string.
    existing_functions = [create_function_definition(action_definition, pddl_domain)[0] for action_definition in action_definitions]
    existing_functions_string = '\n'.join(existing_functions)
    test_string = f"""{existing_functions_string}\n {function_call_string}"""
    return test_string


def attempt_execute_policy(policy, action_definitions, pddl_domain):
    # Define all of the previous functions.
    # Is this an action definition?
    if policy[pddl.PDDLPlan.PDDL_ACTION_TYPE] == pddl.PDDLPlan.PDDL_ACTION_DEFINITION:
       print(f"Attempting to DEFINE function: {policy['action']}")
       test_string = create_new_function_definition_test_string(action_definitions, policy, pddl_domain)
    elif policy[pddl.PDDLPlan.PDDL_ACTION_TYPE] == pddl.PDDLPlan.PDDL_CALL_ACTION:
        print(f"Attempting to CALL function: {policy['action']}")
        test_string = create_function_call_test_string(action_definitions, policy, pddl_domain)
    else:
        return False

    success = False
    try:
        class ExecutionResult():
            def __init__(self):
                python_function = test_string
                # if(policy[pddl.PDDLPlan.PDDL_ACTION_TYPE] == pddl.PDDLPlan.PDDL_CALL_ACTION):
                #     import pdb; pdb.set_trace()
                exec(python_function)
        ex_result = ExecutionResult()
        print(f"Execution result: { ex_result.success}")
        return ex_result.success
    except:
        print("Failed to execute the function defined in code.")
        return success

