#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
import sys
import argparse
import llm_operators.codex as codex
import llm_operators.datasets as datasets
import llm_operators.experiment_utils as experiment_utils
import llm_operators.datasets.crafting_world as crafting_world
from llm_operators.datasets.crafting_world_gen.utils import pascal_to_underline
from llm_operators.datasets.crafting_world import CraftingWorld20230204Simulator, local_search_for_subgoal, SimpleConjunction

# Import Jacinle.
JACINLE_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "../jacinle")
print("Adding jacinle path: {}".format(JACINLE_PATH))
sys.path.insert(0, JACINLE_PATH)

# Import Concepts.
CONCEPTS_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "../concepts")
print("Adding concepts path: {}".format(CONCEPTS_PATH))
sys.path.insert(0, CONCEPTS_PATH)

crafting_world.SKIP_CRAFTING_LOCATION_CHECK = True
print('Skipping location check in Crafting World.')

import concepts.pdsketch as pds
from concepts.pdsketch.strips.strips_grounding_onthefly import OnTheFlyGStripsProblem, ogstrips_bind_arguments

ALL = 'ALL'


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pddl_domain_name", type=str, help="Name of the PDDL domain to load.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset of planning problems to load.")
    parser.add_argument("--dataset_fraction", default=1.0, type=float, help="Fraction of the overall dataset to work with. Lower than 1.0 for debugging purposes only.")
    parser.add_argument("--dataset_pddl_directory", type=str, help="Location of the top level PDDL directory.")

    # Mark the fraction of the dataset that we want to supervise on.
    parser.add_argument("--goal_supervision_fraction", default=0.1, type=float, help="Randomly selected fraction of the dataset to supervise with ground truth PDDL goals.")
    parser.add_argument("--initial_goal_supervision_prefix", type=str, nargs="+", default=[ALL], help="Which initial goal types to supervise on, or ALL if we want to sample ALL of the goal types. This will be sampled in accordance to the underlying distribution of problems of that problem type.")
    parser.add_argument("--plan_supervision_fraction", default=0.1, type=float, help="Randomly selected fraction of the dataset to supervise with ground truth PDDL goals.")
    parser.add_argument("--initial_plans_prefix", type=str, nargs="+", default=[ALL], help="Which initial plan types to supervise on. Used to seed the Codex proposals, or ALL if we want some subset of the initial")

    # Codex proposal parameters.
    parser.add_argument("--initial_pddl_predicates", type=str, nargs="+", default=[], help="Which initial PDDL predicates to run with.  Used to seed the Codex proposals.")
    parser.add_argument("--initial_pddl_operators", type=str, nargs="+", help="Which initial PDDL operators to run with.  Used to seed the Codex proposals.")

    parser.add_argument("--verbose", action="store_true", help="Whether to print out debugging information.")

    args = parser.parse_args()

    experiment_utils.hook_exception_ipdb()

    pddl_domain = datasets.load_pddl_domain(args.pddl_domain_name, args.initial_pddl_operators, args.verbose)
    # pddl_domain.init_operators_to_scores(args.operator_pseudocounts)

    # Load planning dataset.
    planning_problems = datasets.load_planning_problems_dataset(
        dataset_name=args.dataset_name,
        dataset_fraction=args.dataset_fraction,
        dataset_pddl_directory=args.dataset_pddl_directory,
        initial_goal_supervision_fraction=args.goal_supervision_fraction,
        initial_goal_supervision_prefix=args.initial_goal_supervision_prefix,
        initial_plan_supervision_fraction=args.plan_supervision_fraction,
        initial_plan_supervision_prefix=args.initial_plans_prefix,
        initial_pddl_operators=args.initial_pddl_operators,
        domain=pddl_domain,
        verbose=args.verbose,
    )

    current_domain_string = pddl_domain.to_string(
        ground_truth_operators=False,
        current_operators=True,
    )
    pds_domain = pds.load_domain_string(current_domain_string)

    # Baseline 0: LLM(nl_goal) -> pddl_goal; motion_planner(pddl_goal) -> primitive_plan
    # run_brute_force_search(pds_domain, planning_problems['train'])
    # Baseline 1: LLM(nl_goal) -> primitive_plan
    # run_manual_solution_primitive(pds_domain, planning_problems['train'])
    # Baseline 2: LLM(nl_goal) -> subgoal_sequence; motion_planner(subgoal_sequence) -> primitive_plan
    # run_manual_solution_subgoal(pds_domain, planning_problems['train'])
    # Baseline 3 (Voyager): LLM(nl_goal) -> high_level_policy; LLM(high_level_policy) -> low_level_policy
    run_policy(pds_domain, planning_problems['train'])


def load_state_from_problem(pds_domain, problem_record, pddl_goal=None):
    pddl_problem = problem_record.ground_truth_pddl_problem
    current_problem_string = pddl_problem.get_ground_truth_pddl_string()
    problem = pds.load_problem_string(current_problem_string, pds_domain, return_tensor_state=False)

    gproblem = OnTheFlyGStripsProblem.from_domain_and_problem(pds_domain, problem)
    simulator = CraftingWorld20230204Simulator()
    simulator.reset_from_state(gproblem.objects, gproblem.initial_state)

    gt_goal = [x[1:-1] for x in pddl_problem.ground_truth_goal_list]

    return simulator, gt_goal


def run_manual_solution_primitive(pds_domain, problems):
    for problem_key, problem in problems.items():
        print('Now solving problem: {}'.format(problem_key))
        simulator, gt_goal = load_state_from_problem(pds_domain, problem)

        succ = True
        # NB(Jiayuan Mao @ 2023/09/18): problem.ground_truth_primitive_plan should be proposed by LLM given the NL goal.
        for action in problem.ground_truth_primitive_plan:
            print('  Action: {}'.format(action))
            rv = getattr(simulator, action['action'])(*action['args'])
            if not rv:
                print('    Failed to execute action: {}'.format(action))
                succ = False
                break

        if succ:
            succ = simulator.goal_satisfied(gt_goal)

        if not succ:
            print('  Failed to solve problem: {}'.format(problem_key))
            print('  Goal: {}'.format(gt_goal))
            from IPython import embed; embed();
            raise ValueError()

        print('Success: {}'.format(succ))


def run_manual_solution_subgoal(pds_domain, problems):
    # Run manual solutions proposed by LLMs.
    for problem_key, problem in problems.items():
        print('Now solving problem: {}'.format(problem_key))
        simulator, gt_goal = load_state_from_problem(pds_domain, problem)

        succ = True
        # NB(Jiayuan Mao @ 2023/09/18): problem.ground_truth_subgoal_sequence should be proposed by LLM given the NL goal.
        for subgoal in problem.ground_truth_subgoal_sequence:
            print('  Subgoal: {}'.format(subgoal))
            rv = local_search_for_subgoal(simulator, SimpleConjunction(subgoal))
            if rv is None:
                print('    Failed to achieve subgoal: {}'.format(subgoal))
                succ = False
                break
            simulator, _ = rv

        if succ:
            succ = simulator.goal_satisfied(gt_goal)

        if not succ:
            print('  Failed to solve problem: {}'.format(problem_key))
            print('  Goal: {}'.format(gt_goal))
            from IPython import embed; embed();
            raise ValueError()

        print('Success: {}'.format(succ))


def run_brute_force_search(pds_domain, problems):
    for problem_key, problem in problems.items():
        print('Now solving problem: {}'.format(problem_key))
        simulator, gt_goal = load_state_from_problem(pds_domain, problem)

        # NB(Jiayuan Mao @ 2023/09/18): gt_goal should be proposed by LLM given the NL goal.
        rv = local_search_for_subgoal(simulator, SimpleConjunction(gt_goal))
        if rv is None:
            print('  Failed to solve problem: {}'.format(problem_key))
            print('  Goal: {}'.format(gt_goal))
        else:
            simulator, _ = rv
            succ = simulator.goal_satisfied(gt_goal)
            print('Success: {}'.format(succ))


def run_policy(pds_domain, problems):
    import llm_operators.datasets.crafting_world_skill_lib as skill_lib

    for problem_key, problem in problems.items():
        print('Now solving problem: {}'.format(problem_key))
        simulator, gt_goal = load_state_from_problem(pds_domain, problem)

        # NB(Jiayuan Mao @ 2023/09/18): The following lines use rules to generate the high-level policy.
        # This should be replaced by the LLM model.
        target_inventory = None
        target_object = None
        target_object_type = None
        for item in gt_goal:
            parts = item.split()
            if parts[0] == 'inventory-holding':
                target_inventory = int(parts[1][1:])
                target_object = parts[2]
            elif parts[0] == 'object-of-type':
                target_object_type = pascal_to_underline(parts[2])

        if target_inventory is None:
            raise ValueError('Could not find target inventory in goal: {}'.format(gt_goal))
        if target_object is None:
            raise ValueError('Could not find target object in goal: {}'.format(gt_goal))
        if target_object_type is None:
            raise ValueError('Could not find target object type in goal: {}'.format(gt_goal))

        # More functions in skill_lib should be generated by LLMs.
        if target_object_type == 'wood':
            skill_lib.mine_wood(simulator, target_inventory, target_object)
        elif target_object_type == 'potato':
            skill_lib.mine_potato(simulator, target_inventory, target_object)
        else:
            print('SKIP: not implemented for goal {}'.format(gt_goal))
            continue

        rv = simulator.goal_satisfied(gt_goal)
        if not rv:
            print('  Failed to solve problem: {}'.format(problem_key))
            print('  Goal: {}'.format(gt_goal))
            from IPython import embed; embed();
            raise ValueError()
        else:
            print('Success: {}'.format(rv))


if __name__ == '__main__':
    main()

