from collections import defaultdict
import random
import json
import pddl
from typing import Sequence, Dict

import numpy as np
from pddl import Domain, OtherDomain, PDDLPlan

PDDL_PLAN = "pddl_plan"
PDDL_PLAN_OVERALL_COST = "pddl_plan_overall_cost"


class Problem:
    # A planning problem, which contains both PDDL and low-level plan information.
    def __init__(
        self,
        problem_id=None,
        dataset_split=None,
        language=None,
        ground_truth_pddl_plan=None,
        ground_truth_pddl_problem=None,
        should_supervise_pddl=False,
        goal_prefix=None,
    ):
        self.problem_id = problem_id
        self.dataset_split = dataset_split
        self.goal_prefix = goal_prefix

        self.language = language  # An NL string describing the planning problem.
        self.ground_truth_pddl_problem = (
            ground_truth_pddl_problem  # Ground truth PDDL problem object.
        )
        self.constants_in_problem_file = False  # Flag for only the ALFRED domain, which includes constants defined in the problem files not the

        self.ground_truth_pddl_plan = None
        if ground_truth_pddl_plan is not None:
            if isinstance(ground_truth_pddl_plan, PDDLPlan):
                self.ground_truth_pddl_plan = ground_truth_pddl_plan
            elif isinstance(ground_truth_pddl_plan, str):
                self.ground_truth_pddl_plan = PDDLPlan(
                    plan_string=ground_truth_pddl_plan
                )
            else:
                self.ground_truth_pddl_plan = PDDLPlan(
                    plan=ground_truth_pddl_plan
                )  # A ground truth PDDLPlan object.

        self.should_supervise_pddl = (
            should_supervise_pddl  # Whether to include the PDDL in initial supervision
        )
        # One or more proposed PDDL goals.
        self.codex_raw_goals = []
        self.proposed_pddl_goals = []
        # One or more proposed plans. Array of PDDL {action, args} operator sequences.
        self.proposed_pddl_plans = []

        # Evaluated PDDL plans that solve proposed_pddl_goals, created by a task planner. This is reset at each iteration.
        # This is a dict from {goal : set(PDDLPlan)}
        self.evaluated_pddl_plans = defaultdict(set)

        # This is a dict to all of the {(goal, PDDLPlan) : MotionPlanResult}, created by a motion planner. These may be successful or failed.
        # This is reset at each iteration.
        self.evaluated_motion_planner_results = dict()

        # This contains any solved motion plan results, which are stored completely with their goal, PDDL plan object, and motion plan.
        self.solved_motion_plan_results = dict()

    def get_evaluated_pddl_plan_json(self):
        return {
            "file_name": self.problem_id,
            "plans": [
                {"goal": g, "plan": pddl_plan.plan}
                for g in self.evaluated_pddl_plans
                for pddl_plan in self.evaluated_pddl_plans[g]
            ],
        }

    def get_evaluated_motion_plan_json(self):
        return {
            "file_name": self.problem_id,
            "motion_plans": [
                {
                    "goal": g,
                    "plan": pddl_plan_string,
                    "task_success": result.task_success,
                    "last_failed_operator": result.last_failed_operator,
                    "last_failed_predicate": result.last_failed_predicate,
                }
                for (
                    (g, pddl_plan_string),
                    result,
                ) in self.evaluated_motion_planner_results.items()
            ],
        }

    def reset_evaluated_pddl_plans(self):
        self.evaluated_pddl_plans = defaultdict(set)

    def reset_evaluated_motion_planner_results(self):
        self.evaluated_pddl_plans = dict()

    def update_solved_motion_plan_results(self):
        for k in self.evaluated_motion_planner_results:
            if self.evaluated_motion_planner_results[k].task_success:
                self.solved_motion_plan_results[
                    k
                ] = self.evaluated_motion_planner_results[k]

    def update_evaluated_pddl_plans(self, new_evaluated_pddl_plans):
        for g in new_evaluated_pddl_plans:
            self.evaluated_pddl_plans[g].update(new_evaluated_pddl_plans[g])

    def get_highest_likelihood_evaluated_pddl_plan(self):
        """Returns the best evaluated PDDL plan, or None if no plans have been evaluated."""
        # TODO: right now we only propose one plan anyway, so just return it.
        print("TODO: LCW - implement this for choosing plans.")
        for goal in self.evaluated_pddl_plans:
            return self.evaluated_pddl_plans[goal]

    def __repr__(self):
        return (
            f"Problem(\n"
            f"problem_id={self.problem_id},\n"
            f'language="{self.language}",\n'
            f'goal_prefix="{self.goal_prefix}",\n'
            f"ground_truth_pddl_plan={self.ground_truth_pddl_plan},\n"
            f"should_supervise_pddl={self.should_supervise_pddl}\n"
            f"proposed_pddl_goals = {self.proposed_pddl_goals}\n"
            f"proposed_pddl_plans = {self.proposed_pddl_plans}\n)"
        )
    
    def get_preconditions(self):
        return pddl.PDDLParser._parse_problem_predicates(self.ground_truth_pddl_problem.ground_truth_pddl_problem_string)


def load_pddl_supervision(
    supervision_name: str, verbose: bool = False
) -> Dict[str, str]:
    """Supervision is a list of PDDL domains to teach Codex "basic grammar" of PDDL.

    Args:
        supervision_name: identifier for the supervision.

    Returns:
        dict from domain name to supervision PDDL strings.
    """
    # Load the supervision goals and operators. These should be in dataset/<supervision_name>-NLgoals-operators.json
    if supervision_name is None:
        return []
    with open(f"data/dataset/{supervision_name}-NLgoals-operators.json") as f:
        pddl_supervision = {goal["domain_file"]: goal for goal in json.load(f)}

    # Load natural language + objects for these.  These should be in dataset/<supervision_name>-NL.json
    with open(f"data/dataset/{supervision_name}-NL.json") as f:
        for nl_goal in json.load(f):
            if nl_goal["domain_file"] in pddl_supervision:
                pddl_supervision[nl_goal["domain_file"]]["NL_goal"] = nl_goal["NL_goal"]
                pddl_supervision[nl_goal["domain_file"]]["object_list"] = nl_goal[
                    "object_list"
                ]
                with open(nl_goal["domain_file"]) as j:
                    pddl_supervision[nl_goal["domain_file"]]["domain"] = OtherDomain(
                        j.read()
                    )
                with open(pddl_supervision[nl_goal["domain_file"]]["file_name"]) as g:
                    pddl_supervision[nl_goal["domain_file"]][
                        "pddl_problem_string"
                    ] = g.read()

    for domain_file in list(pddl_supervision.keys()):
        if "NL_goal" not in pddl_supervision[domain_file]:
            del pddl_supervision[domain_file]
    if verbose:
        print("load_pddl_supervision from the following domain files:")
        for domain_file in pddl_supervision:
            print(domain_file)

    return pddl_supervision


# Planning domains available for the --pddl_domain_name flag.
PLANNING_PDDL_DOMAINS_REGISTRY = dict()


def register_planning_pddl_domain(name):
    def wrapper(f):
        PLANNING_PDDL_DOMAINS_REGISTRY[name] = f
        return f

    return wrapper


def load_pddl_domain(
    pddl_domain_name: str, initial_pddl_operators: Sequence[str], verbose: bool = False
) -> Domain:
    """Main entry for loading a PDDL domain.

    Args:

    """
    planning_domain_loader = PLANNING_PDDL_DOMAINS_REGISTRY[pddl_domain_name]
    pddl_domain = planning_domain_loader(verbose)
    # Only keep the designated initial operators.
    for o in list(pddl_domain.operators.keys()):
        if o not in initial_pddl_operators:
            pddl_domain.remove_operator(o)
    if verbose:
        print("\nInitializing with operators: ")
        for o in list(pddl_domain.operators.keys()):
            print(o)
    return pddl_domain


# Planning problem datasets available for the --dataset_name flag.
PLANNING_PROBLEMS_REGISTRY = dict()


def register_planning_domain_problems(name):
    def wrapper(f):
        PLANNING_PROBLEMS_REGISTRY[name] = f
        return f

    return wrapper


def get_problem_ids_with_ground_truth_operators(
    initial_pddl_operators, planning_dataset, split="train"
):
    """
    :ret: list of problem IDs where the ground truth plans contain these operators.
    """
    problem_ids = []
    initial_pddl_operators = set(initial_pddl_operators)
    for problem_id in planning_dataset[split]:
        problem = planning_dataset[split][problem_id]
        ground_truth_operators = set(
            [
                operator[PDDLPlan.PDDL_ACTION]
                for operator in problem.ground_truth_pddl_plan.plan
            ]
        )
        # Check that this plan doesn't contain any more operators than the initial ones.
        if len(initial_pddl_operators.union(ground_truth_operators)) <= len(
            initial_pddl_operators
        ):
            problem_ids.append(problem_id)
    return problem_ids


def get_problem_ids_with_initial_plans_prefix(
    initial_plans_prefix, planning_dataset, split="train"
):
    """
    :ret: list of problem IDs where the ground truth plans contain these operators.
    """
    problem_ids = []
    for problem_id in planning_dataset[split]:
        problem = planning_dataset[split][problem_id]
        if problem.goal_prefix in initial_plans_prefix:
            problem_ids.append(problem_id)
    return problem_ids


def load_planning_problems_dataset(
    dataset_name,
    dataset_fraction,
    training_plans_fraction,
    dataset_pddl_directory,
    initial_pddl_operators,
    initial_plans_prefix=None,
    verbose=False,
):
    planning_domain_loader = PLANNING_PROBLEMS_REGISTRY[dataset_name]
    # Load some fraction of the dataset.

    planning_dataset = planning_domain_loader(
        dataset_pddl_directory=dataset_pddl_directory,
        dataset_fraction=dataset_fraction,
        verbose=verbose,
    )

    if initial_plans_prefix:
        candidate_training_plans = get_problem_ids_with_initial_plans_prefix(
            initial_plans_prefix, planning_dataset, split="train"
        )
    else:
        # Get candidate problems to supervise on.
        candidate_training_plans = get_problem_ids_with_ground_truth_operators(
            initial_pddl_operators, planning_dataset, split="train"
        )
    # Supervise on a maximum of the training plans fraction.
    num_to_supervise = min(
        int(np.ceil(training_plans_fraction * len(planning_dataset["train"]))),
        len(candidate_training_plans),
    )
    problems_to_supervise = random.sample(candidate_training_plans, num_to_supervise,)
    if verbose:
        print("Supervising on these problems: ")
        print(problems_to_supervise)
    for problem_id in problems_to_supervise:
        planning_dataset["train"][problem_id].should_supervise_pddl = True

    if verbose:
        print(f"training_plans_fraction: {training_plans_fraction}")
        print(f"supervising on: {num_to_supervise} problems.")
    return planning_dataset

