"""
main.py

Usage:
    # Load a debug fraction of the ALFRED dataset.
    python main.py --experiment_name alfred_linearized_100 --dataset_name alfred_linearized_100 --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_linearized_pddl --output_directory generated/test_outputs

    # Append these flags if you want to mock out the Codex proposals with previous checkpoints.
    --debug_mock_propose_plans --debug_mock_propose_operators --debug_mock_propose_goals
    # Append this flag if you want to mock out the task planner with previous plans.
    --debug_mock_task_plans
"""
import os.path as osp
import sys


ALFRED_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "alfred")
print("Adding ALFRED path: {}".format(ALFRED_PATH))
sys.path.insert(0, ALFRED_PATH)

# Import concepts.
try:

    JACINLE_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "../jacinle")
    print("Adding jacinle path: {}".format(JACINLE_PATH))
    sys.path.insert(0, JACINLE_PATH)
    CONCEPTS_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "../concepts")
    print("Adding concepts path: {}".format(CONCEPTS_PATH))
    sys.path.insert(0, CONCEPTS_PATH)

except:
    pass

try:
    import jacinle

    jacinle.hook_exception_ipdb()
except ImportError:
    # If Jacinle is not installed, that's fine.
    pass

import argparse
import random
import llm_operators.codex as codex
import llm_operators.datasets as datasets
import llm_operators.task_planner as task_planner
import llm_operators.motion_planner as motion_planner
import llm_operators.pddl as pddl
import llm_operators.experiment_utils as experiment_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_name",
    type=str,
    default="",
    help="Experiment name tag. This will be appended to any checkpointed data.",
)
parser.add_argument(
    "--dataset_name", type=str, help="Name of the dataset of planning problems to load."
)
parser.add_argument(
    "--dataset_fraction",
    default=1.0,
    type=float,
    help="Fraction of the overall dataset to work with. Lower than 1.0 for debugging purposes",
)
parser.add_argument(
    "--training_plans_fraction",
    default=1.0,
    type=float,
    help="Fraction of the training problems to initialize with plans. Used to seed the Codex proposals.",
)
parser.add_argument(
    "--dataset_pddl_directory",
    type=str,
    help="Location of the top level PDDL directory.",
)
parser.add_argument(
    "--pddl_domain_name", type=str, help="Name of the PDDL domain to load.",
)

parser.add_argument(
    "--train_iterations", type=int, help="How many training iterations to run."
)
parser.add_argument(
    "--supervision_name",
    type=str,
    default="supervision",
    help="Tag for the supervision dataset to load.",
)

parser.add_argument(
    "--initial_plans_prefix",
    type=str,
    nargs="+",
    help="Which initial plan types to supervise on. Used to seed the Codex proposals",
)

parser.add_argument(
    "--initial_pddl_operators",
    type=str,
    nargs="+",
    help="Which initial PDDL operators to run with.  Used to seed the Codex proposals.",
)
parser.add_argument(
    "--initial_pddl_predicates",
    type=str,
    nargs="+",
    default=[],
    help="Which initial PDDL predicates to run with.  Used to seed the Codex proposals.",
)
parser.add_argument(
    "--operator_propose_minimum_usage",
    type=int,
    default=2,
    help="Minimum number of times an operator must be used to be considered for proposal.",
)
parser.add_argument(
    "--goal_propose_include_codex_types",
    action="store_true",
    help="Whether to include Codex types in the prompts for goal proposal.",
)
parser.add_argument(
    "--planner", type=str, default="task_planner_fd", help="Which planner to use.",
)
parser.add_argument(
    "--output_directory",
    type=str,
    help="Location of the directory for writing outputs.",
)
parser.add_argument("--verbose", action="store_true", help="Run on verbose.")
parser.add_argument(
    "--debug_no_propose_plans_operators_goals",
    action="store_true",
    help="debug: don't run propose_plans_operators_goals. Instead, use ground truths.",
)
parser.add_argument(
    "--debug_mock_propose_goals",
    action="store_true",
    help="debug: mock out goal_proposal.",
)
parser.add_argument(
    "--debug_mock_propose_plans",
    action="store_true",
    help="debug: mock out plan proposal.",
)
parser.add_argument(
    "--debug_mock_propose_operators",
    action="store_true",
    help="debug: mock out operator_proposal.",
)
parser.add_argument(
    "--debug_skip_task_plans",
    action="store_true",
    help="debug: skip task plan grounded search and assume that all of the task plans succeeded.",
)
parser.add_argument(
    "--debug_mock_task_plans",
    action="store_true",
    help="debug: mock out task plan symbolic search.",
)
parser.add_argument(
    "--debug_mock_motion_plans",
    action="store_true",
    help="debug: mock out motion plan grounded search.",
)
parser.add_argument(
    "--debug_skip_motion_plans",
    action="store_true",
    help="debug: skip motion plan grounded search and assume that all of the task plans succeeded.",
)

parser.add_argument(
    "--assume_alfred_teleportation",
    action="store_true",
    help="debug: assume that we never need to directly go to any location..",
)

parser.add_argument(
    "--debug_ground_truth_operators",
    action="store_true",
    help="debug: use ground_truth_operators.",
)
parser.add_argument(
    "--debug_ground_truth_goals",
    action="store_true",
    help="debug: use ground_truth_goals.",
)
parser.add_argument(
    "--debug_stop_after_first_proposal",
    action="store_true",
    help="debug: stop after the first proposal for goals, plans, and operators (no evaluation).",
)

parser.add_argument(
    "--codex_goal_temperature",
    type=float,
    default=codex.DEFAULT_GOAL_TEMPERATURE,
    help="OpenAI temperature for goal proposal.",
)
parser.add_argument(
    "--n_attempts_to_plan",
    type=int,
    default=4,
    help="Number of attempts to iterate over the task and motion planning loop. Starts by planning with all of the operators, from there downsamples.",
)
parser.add_argument(
    "--maximum_operator_arity",
    type=int,
    default=4,
    help="Maximum arity for proposed operators.",
)


def main():
    random.seed(0)

    args = parser.parse_args()

    ###### Initialization. This initializes a set of goals (planning dataset), and a planning domain (a set of predicates + a partial set of initial operators.)
    # Load planning dataset.
    planning_problems = datasets.load_planning_problems_dataset(
        dataset_name=args.dataset_name,
        dataset_fraction=args.dataset_fraction,
        training_plans_fraction=args.training_plans_fraction,
        dataset_pddl_directory=args.dataset_pddl_directory,
        initial_pddl_operators=args.initial_pddl_operators,
        initial_plans_prefix=args.initial_plans_prefix,
        verbose=args.verbose,
    )

    pddl_domain = datasets.load_pddl_domain(
        args.pddl_domain_name, args.initial_pddl_operators, args.verbose
    )

    # Load any external supervision on PDDL domains.
    supervision_pddl = datasets.load_pddl_supervision(
        supervision_name=args.supervision_name, verbose=args.verbose,
    )

    for curr_iteration in range(args.train_iterations):
        output_directory = experiment_utils.get_output_directory(
            curr_iteration=curr_iteration,
            command_args=args,
            experiment_name_to_load=args.experiment_name,
        )
        ###################### Operator sampling.
        # Given a domain and a set of goals, this uses Codex + preprocessing to sample
        # a set of operator definitions for goals.
        if not args.debug_no_propose_plans_operators_goals:
            #### Goal proposal and preprocessing.
            codex.propose_goals_for_problems(
                problems=planning_problems["train"],
                current_domain=pddl_domain,
                output_directory=output_directory,
                supervision_pddl=supervision_pddl,
                verbose=args.verbose,
                temperature=args.codex_goal_temperature,
                initial_pddl_predicates=args.initial_pddl_predicates,
                experiment_name=args.experiment_name,
                use_mock=args.debug_mock_propose_goals,
                use_gt=args.debug_ground_truth_goals,
            )
            pddl.preprocess_goals(
                problems=planning_problems["train"],
                pddl_domain=pddl_domain,
                output_directory=output_directory,
                command_args=args,
                verbose=args.verbose,
            )
            codex.propose_plans_operators_for_problems(
                current_domain=pddl_domain,
                problems=planning_problems["train"],
                supervision_pddl=supervision_pddl,
                n_samples=1,
                minimum_usage=args.operator_propose_minimum_usage,
                verbose=args.verbose,
                output_directory=output_directory,
                command_args=args,
                use_gt=args.debug_ground_truth_operators,
            )
            # TODO (LCW) - this removes the partially grounded (receptacleType ?r FridgeType) rn.
            pddl.preprocess_operators(
                pddl_domain,
                maximum_operator_arity=args.maximum_operator_arity,
                verbose=args.verbose,
                output_directory=output_directory,
                command_args=args,
            )

        if args.debug_stop_after_first_proposal:
            break

        ###################### Refine operators.
        for problem_idx, problem_id in enumerate(planning_problems["train"]):
            if problem_idx == 1:  # TODO (LCW) REMOVE
                should_continue_attempts = True
                for plan_attempt_idx in range(args.n_attempts_to_plan):
                    if should_continue_attempts:
                        # Task plan. Attempts to generate a task plan for each problem.
                        task_planner.attempt_task_plan_for_problem(
                            pddl_domain=pddl_domain,
                            problem_idx=problem_idx,
                            problem_id=problem_id,
                            problems=planning_problems["train"],
                            verbose=args.verbose,
                            command_args=args,
                            output_directory=output_directory,
                            debug_skip=args.debug_skip_task_plans,
                            use_mock=args.debug_mock_task_plans,
                            plan_attempt_idx=plan_attempt_idx,
                        )
                        # Motion plan. Attempts to generate a motion plan for a problem.
                        motion_planner.attempt_motion_plan_for_problem(
                            pddl_domain=pddl_domain,
                            problem_idx=problem_idx,
                            problem_id=problem_id,
                            problems=planning_problems["train"],
                            verbose=args.verbose,
                            command_args=args,
                            output_directory=output_directory,
                            debug_skip=args.debug_skip_motion_plans,
                            use_mock=args.debug_mock_motion_plans,
                            plan_attempt_idx=plan_attempt_idx,
                            dataset_name=args.dataset_name,
                        )
                        should_continue_attempts = pddl.update_pddl_domain_and_problem(
                            pddl_domain=pddl_domain,
                            problem_idx=problem_idx,
                            problem_id=problem_id,
                            problems=planning_problems["train"],
                            verbose=args.verbose,
                            command_args=args,
                        )

        ###################### Finalize and checkpoint iteration.
        pddl.checkpoint_and_reset_operators(
            curr_iteration=curr_iteration,
            pddl_domain=pddl_domain,
            command_args=args,
            output_directory=output_directory,
        )
        pddl.checkpoint_and_reset_plans(
            curr_iteration=curr_iteration,
            pddl_domain=pddl_domain,
            problems=planning_problems["train"],
            command_args=args,
            output_directory=output_directory,
        )

        # Output a final iteration summary.
        experiment_utils.output_iteration_summary(
            curr_iteration=curr_iteration,
            pddl_domain=pddl_domain,
            problems=planning_problems["train"],
            command_args=args,
            output_directory=output_directory,
        )


if __name__ == "__main__":
    main()
