"""
main.py

Usage:
    # Load a debug fraction of the ALFRED dataset.
    python main.py --experiment_name alfred_linearized_100_ground_truth_652023 --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple pick_clean_then_place_in_recep --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject --train_iterations 1 --dataset_pddl_directory data/dataset/alfred_linearized_pddl --output_directory generated --debug_ground_truth_operators --debug_ground_truth_goals --n_attempts_to_plan 4 --motionplan_search_type counter

    # Append these flags if you want to mock out the Codex proposals with previous checkpoints.
    --debug_mock_propose_plans --debug_mock_propose_operators --debug_mock_propose_goals
    # Append this flag if you want to mock out the task planner with previous plans.
    --debug_mock_task_plans
"""

import os
import os.path as osp
import sys
import numpy as np

ALL = "ALL"

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
parser.add_argument("--dataset_name", type=str, help="Name of the dataset of planning problems to load.")
parser.add_argument(
    "--dataset_fraction",
    default=1.0,
    type=float,
    help="Fraction of the overall dataset to work with. Lower than 1.0 for debugging purposes only.",
)
parser.add_argument(
    "--dataset_pddl_directory",
    type=str,
    help="Location of the top level PDDL directory.",
)
parser.add_argument(
    "--pddl_domain_name",
    type=str,
    help="Name of the PDDL domain to load.",
)

parser.add_argument("--train_iterations", type=int, help="How many training iterations to run.")
parser.add_argument(
    "--supervision_name",
    type=str,
    default="supervision",
    help="Tag for the supervision dataset to load.",
)

parser.add_argument(
    "--goal_supervision_fraction",
    default=0.1,
    type=float,
    help="Randomly selected fraction of the dataset to supervise with ground truth PDDL goals.",
)

parser.add_argument(
    "--initial_goal_supervision_prefix",
    type=str,
    nargs="+",
    default=[ALL],
    help="Which initial goal types to supervise on, or ALL if we want to sample ALL of the goal types. This will be sampled in accordance to the underlying distribution of problems of that problem type.",
)

parser.add_argument(
    "--plan_supervision_fraction",
    default=0.1,
    type=float,
    help="Randomly selected fraction of the dataset to supervise with ground truth PDDL goals.",
)

parser.add_argument(
    "--initial_plans_prefix",
    type=str,
    nargs="+",
    help="Which initial plan types to supervise on. Used to seed the Codex proposals, or ALL if we want some subset of the initial",
)
parser.add_argument(
    "--external_plan_supervision",
    type=str,
    default=None,
    help="If provided, file containing initial plans that will be provided as supervision.",
)
parser.add_argument(
    "--external_operator_supervision",
    type=str,
    default=None,
    help="If provided, file HEADER path containing the supervision used for external operators. Assuming you are prompting GPT-3.5, this should be followed by two file suffices, one _system.txt and the other _user.txt.",
)
parser.add_argument(
    "--external_operator_sample_with_prompt",
    action="store_true",
    help="If provided, this assumes that instead of taking N discrete samples with the same prompt, we will 'sample' the LLM based on the user message, and attempt to parse out discrete operators from the prompt itself.",
)
parser.add_argument(
    "--external_operator_names",
    type=str,
    nargs="+",
    help="Initial PDDL operators that were provided. These will be excluded from downstream proposal.",
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
    "--planner",
    type=str,
    default="task_planner_fd",
    help="Which planner to use.",
)
parser.add_argument(
    "--output_directory",
    type=str,
    help="Location of the directory for writing outputs.",
)
parser.add_argument("--verbose", action="store_true", help="Run on verbose.")
parser.add_argument(
    "--debug_export_failed_pddl", type=str, default=None, help="Export failed PDDL problems to this directory."
)
parser.add_argument(
    "--debug_no_propose_plans_operators_goals",
    action="store_true",
    help="debug: don't run propose_plans_operators_goals. Instead, use ground truths.",
)
parser.add_argument(
    "--debug_mock_propose_goals",
    action="store_true",
    help="debug: mock out goal_proposal. If not, starts over.",
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
    "--debug_start_problem_idx",
    type=int,
    default=0,
    help="debug: start at this problem index.",
)
parser.add_argument(
    "--debug_skip_problems",
    type=int,
    nargs="+",
    help="debug: skip these problems.",
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
    "--codex_operator_temperature",
    type=float,
    default=codex.DEFAULT_OPERATOR_TEMPERATURE,
    help="OpenAI temperature for goal proposal.",
)
parser.add_argument(
    "--n_goal_samples",
    type=int,
    default=4,
    help="Number of initial samples to take from the LLM for goals.",
)
parser.add_argument(
    "--n_plan_samples",
    type=int,
    default=5,
    help="Number of initial samples to take from the LLM for plans.",
)
parser.add_argument(
    "--n_operator_samples",
    type=int,
    default=3,
    help="Number of initial samples to take from the LLM for operators.",
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

parser.add_argument(
    "--motionplan_search_type",
    type=str,
    default="bfs",
    help="Which search type to use for motion planning: supports bfs or counter",
)

parser.add_argument(
    "--random_seed",
    type=int,
    default=0,
    help="Random seed for replication.",
)
parser.add_argument(
    "--checkpoint_every_n_problem_plans", type=int, default=2, help="Write out results every n problems."
)
parser.add_argument(
    "--minimum_n_operators",
    type=int,
    default=10,
    help="Minimum number of operators we can sample in a proposed library at any point.",
)
parser.add_argument(
    "--operator_pseudocount_success",
    type=int,
    default=0.5,
    help="Assume each operator succeeded this percentage of the pseudocount times (MAP smoothing)",
)
parser.add_argument(
    "--operator_pseudocounts",
    type=int,
    default=0.1,
    help="Assume we've tried each operator at least this many times (MAP smoothing)",
)
parser.add_argument(
    "--operator_acceptance_threshold",
    type=float,
    default=0.5,
    help="After each iteration, we prune out operators that have less than this probability of success.",
)


def main():
    args = parser.parse_args()
    random.seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)
    ###### Log all of the arguments that we ran this experiment with and the experiment date.
    experiment_utils.output_experiment_parameters(args)

    ###### Initialization. This initializes a set of goals (planning dataset), and a planning domain (a set of predicates + a partial set of initial operators.)
    # Load planning dataset.
    planning_problems = datasets.load_planning_problems_dataset(
        dataset_name=args.dataset_name,
        dataset_fraction=args.dataset_fraction,
        goal_supervision_fraction=args.goal_supervision_fraction,
        initial_goal_supervision_prefix=args.initial_goal_supervision_prefix,
        dataset_pddl_directory=args.dataset_pddl_directory,
        initial_pddl_operators=args.initial_pddl_operators,
        initial_plans_prefix=args.initial_plans_prefix,
        verbose=args.verbose,
    )

    pddl_domain = datasets.load_pddl_domain(args.pddl_domain_name, args.initial_pddl_operators, args.verbose)
    pddl_domain.init_operators_to_scores(
        operator_pseudocounts=args.operator_pseudocounts,
        operator_pseudocount_success=args.operator_pseudocount_success,
    )

    # Load any external supervision on PDDL domains.
    if args.supervision_name != "None":
        supervision_pddl = datasets.load_pddl_supervision(
            supervision_name=args.supervision_name,
            verbose=args.verbose,
        )
    else:
        supervision_pddl = None

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
                supervision_pddl=None,  # (ZS 7/27/23) skip supervising on external datasets for now.
                verbose=args.verbose,
                n_samples=args.n_goal_samples,
                temperature=args.codex_goal_temperature,
                initial_pddl_predicates=args.initial_pddl_predicates,
                experiment_name=args.experiment_name,
                use_mock=args.debug_mock_propose_goals,
                use_gt=args.debug_ground_truth_goals,
                args=args,
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
                n_plan_samples=args.n_plan_samples,
                n_operator_samples=args.n_operator_samples,
                minimum_usage=args.operator_propose_minimum_usage,
                verbose=args.verbose,
                output_directory=output_directory,
                command_args=args,
                use_gt=args.debug_ground_truth_operators,
                external_plan_supervision=args.external_plan_supervision,
                external_operator_supervision=args.external_operator_supervision,
                external_operator_sample_with_prompt=args.external_operator_sample_with_prompt,
                operator_temperature=args.codex_operator_temperature,
                external_operator_names=args.external_operator_names,
            )
            pddl.preprocess_operators(
                pddl_domain,
                maximum_operator_arity=args.maximum_operator_arity,
                verbose=args.verbose,
                output_directory=output_directory,
                command_args=args,
            )

        if args.debug_stop_after_first_proposal:
            break

        if output_directory:
            if not args.debug_skip_task_plans and not args.debug_mock_task_plans:
                experiment_tag = "" if len(args.experiment_name) < 1 else f"{args.experiment_name}_"
                output_filepath = f"{experiment_tag}task_planner_results.csv"
                output_filename = osp.join(output_directory, output_filepath)

                if osp.exists(output_filename):
                    os.remove(output_filename)

        ###################### Refine operators.
        if args.debug_mock_task_plans:
            pddl.load_operator_checkpoint(
                curr_iteration=curr_iteration,
                pddl_domain=pddl_domain,
                command_args=args,
                output_directory=output_directory,
            )

        for problem_idx, problem_id in enumerate(planning_problems["train"]):
            finished_epoch = problem_idx == len(planning_problems["train"]) - 1
            if problem_idx < args.debug_start_problem_idx or (
                args.debug_skip_problems is not None and problem_idx in args.debug_skip_problems
            ):
                continue
            # Continue on a problem as long as we have not succeeded in finding a motion plan result.
            any_success = False
            for plan_attempt_idx in range(args.n_attempts_to_plan):
                for goal_idx in range(args.n_goal_samples):
                    if any_success:
                        continue
                    else:
                        # Task plan. Attempts to generate a task plan for each problem.
                        found_new_task_plan, new_task_plans = task_planner.attempt_task_plan_for_problem(
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
                            goal_idx=goal_idx,
                            random_generator=rng,
                            minimum_n_operators=args.minimum_n_operators,
                        )
                        if found_new_task_plan:
                            # Motion plan. Attempts to generate a motion plan for a problem.
                            (
                                any_motion_planner_success,
                                new_motion_plan_keys,
                            ) = motion_planner.attempt_motion_plan_for_problem(
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
                                new_task_plans=new_task_plans,
                            )
                            # Update the global operator scores from the problem.
                            pddl.update_pddl_domain_and_problem(
                                pddl_domain=pddl_domain,
                                problem_idx=problem_idx,
                                problem_id=problem_id,
                                problems=planning_problems["train"],
                                verbose=args.verbose,
                                command_args=args,
                                new_motion_plan_keys=new_motion_plan_keys,
                            )
                            if any_motion_planner_success:
                                any_success = True

            # Checkpoint operators, only reset if we're at the end of the iteration.
            if problem_idx % args.checkpoint_every_n_problem_plans == 0 or finished_epoch:
                pddl.checkpoint_and_reset_operators(
                    curr_iteration=curr_iteration,
                    pddl_domain=pddl_domain,
                    command_args=args,
                    output_directory=output_directory,
                    reset_operators=finished_epoch,
                    operator_acceptance_threshold=args.operator_acceptance_threshold,
                    operator_pseudocounts=args.operator_pseudocounts,
                )
                pddl.checkpoint_and_reset_plans(
                    curr_iteration=curr_iteration,
                    pddl_domain=pddl_domain,
                    problems=planning_problems["train"],
                    command_args=args,
                    output_directory=output_directory,
                    reset_plans=finished_epoch,
                )
                # Output the interim iteration summary to logs.
                experiment_utils.output_iteration_summary(
                    curr_iteration=curr_iteration,
                    pddl_domain=pddl_domain,
                    problems=planning_problems["train"],
                    command_args=args,
                    output_directory=output_directory,
                    finished_epoch=finished_epoch,
                    problem_idx=problem_idx,
                    total_problems=len(planning_problems["train"]),
                )


if __name__ == "__main__":
    main()
