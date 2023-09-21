import frozendict
from llm_operators import motion_planner
from llm_operators import pddl

from llm_operators.datasets.crafting_world import CraftingWorld20230204Simulator, local_search_for_subgoal, SimpleConjunction

# Import Concepts.
import os.path as osp
import sys
CONCEPTS_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "../concepts")
print("Adding concepts path: {}".format(CONCEPTS_PATH))
sys.path.insert(0, CONCEPTS_PATH)

import concepts.pdsketch as pds
from concepts.pdsketch.strips.strips_grounding_onthefly import OnTheFlyGStripsProblem, ogstrips_bind_arguments


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
        problem = problems[problem_id]
        simulator, gt_goal = load_state_from_problem(pddl_domain, problem)
        any_motion_planner_success = True
        for idx, subgoal_sequence in enumerate(problems[problem_id].proposed_pddl_task_predicates):
            print(f"Evaluating [{idx+1}/{len(problems[problem_id].proposed_pddl_task_predicates)}] LLM task plans.")
            
            print(f"Ground truth subgoal sequence is: {problem.ground_truth_subgoal_sequence}")
            print(f"LLM proposed subgoal sequence: {subgoal_sequence}")
            for subgoal in problem.ground_truth_subgoal_sequence:
                print('  Subgoal: {}'.format(subgoal))
                rv = local_search_for_subgoal(simulator, SimpleConjunction(subgoal))
                if rv is None:
                    print('    Failed to achieve subgoal: {}'.format(subgoal))
                    succ = False
                    break
                simulator, _ = rv

            if any_motion_planner_success:
                any_motion_planner_success = simulator.goal_satisfied(gt_goal)
            if any_motion_planner_success:
                return any_motion_planner_success

    return any_motion_planner_success


def _run_llm_propose_code_policies_motion_planner(pddl_domain, problem_idx, problem_id, planning_problems, args, curr_iteration, output_directory, plan_pass_identifier, plan_attempt_idx, goal_idx, rng, split):
    problems = planning_problems[split]
    any_motion_planner_success = False
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
