from collections import defaultdict
import random
import json
from typing import Sequence, Dict

import numpy as np
from llm_operators.pddl import Domain, OtherDomain, PDDLPlan

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
        should_supervise_pddl_goal=False,
        should_supervise_pddl_plan=False,
        goal_prefix=None,
        chain_of_thought=None,
    ):
        self.problem_id = problem_id
        self.dataset_split = dataset_split
        self.goal_prefix = goal_prefix
        self.chain_of_thought = chain_of_thought

        self.language = language  # An NL string describing the planning problem.
        self.ground_truth_pddl_problem = ground_truth_pddl_problem  # Ground truth PDDL problem object.
        self.constants_in_problem_file = (
            False  # Flag for only the ALFRED domain, which includes constants defined in the problem files not the
        )

        self.ground_truth_pddl_plan = None
        self.correct_pddl_goal = False  # True if at least one of the proposed PDDL goals is correct.
        if ground_truth_pddl_plan is not None:
            if isinstance(ground_truth_pddl_plan, PDDLPlan):
                self.ground_truth_pddl_plan = ground_truth_pddl_plan
            elif isinstance(ground_truth_pddl_plan, str):
                self.ground_truth_pddl_plan = PDDLPlan(plan_string=ground_truth_pddl_plan)
            else:
                self.ground_truth_pddl_plan = PDDLPlan(plan=ground_truth_pddl_plan)  # A ground truth PDDLPlan object.

        self.should_supervise_pddl_goal = (
            should_supervise_pddl_goal  # Whether to supervise specifically on ground truth information about the goal.
        )

        self.should_supervise_pddl_plan = should_supervise_pddl_plan  # Whether to include the PDDL in initial supervision
        # One or more proposed PDDL goals.
        self.codex_raw_goals = []
        self.proposed_pddl_goals = []
        # One or more proposed plans from an LLM. Array of PDDL {action, args} operator sequences.
        self.proposed_pddl_plans = []

        # One of more propose task predicates for the task predicates baseline.
        self.codex_raw_task_predicates = [] # List of strings 
        self.proposed_pddl_task_predicates = [] # List of preprocessed objects

        # Evaluated PDDL plans that solve proposed_pddl_goals, created by a task planner. This is reset at each iteration.
        # This is a dict from {goal : list(PDDLPlan)} # But the list items are deduped.
        self.evaluated_pddl_plans = defaultdict(list)

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
                    "plan": result.pddl_plan.plan_string,
                    "task_success": result.task_success,
                    "last_failed_operator": result.last_failed_operator,
                    "max_satisfied_predicates": result.max_satisfied_predicates,
                    "total_trajs_sampled": result.total_trajs_sampled,
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
                self.solved_motion_plan_results[k] = self.evaluated_motion_planner_results[k]

    def update_evaluated_pddl_plans(self, new_evaluated_pddl_plans):
        # Return true if we really made an update -- that is, that we really added new plans.
        updated_pddl_plans = False
        for g in new_evaluated_pddl_plans:
            current_pddl_plans = set(self.evaluated_pddl_plans[g])
            new_pddl_plan_for_goal = new_evaluated_pddl_plans[g]
            if new_pddl_plan_for_goal not in current_pddl_plans:
                self.evaluated_pddl_plans[g].append(new_pddl_plan_for_goal)
                updated_pddl_plans = True
        return updated_pddl_plans

    def get_solved_pddl_plan_string(self):
        """Returns one of the solved PDDL plan, or None if no plans have been solved."""
        for goal, pddl_plan in self.solved_motion_plan_results:
            return pddl_plan  # already a string
        raise RuntimeError('No solved PDDL plan found.')

    def get_highest_likelihood_evaluated_pddl_plan(self):
        """Returns the best evaluated PDDL plan, or None if no plans have been evaluated."""
        # TODO(Jiayuan Mao @ 2023/08/30): I don't quite understand the purpose of this function. Maybe we should remove this todo note.
        # Right now I think we are consider all proposed plans? Or should we?
        # TODO: right now we only propose one plan anyway, so just return it.
        # print("TODO: LCW - implement this for choosing plans.")
        for goal in self.evaluated_pddl_plans:
            return self.evaluated_pddl_plans[goal][0]

    def __repr__(self):
        return (
            f"Problem(\n"
            f"problem_id={self.problem_id},\n"
            f'language="{self.language}",\n'
            f'goal_prefix="{self.goal_prefix}",\n'
            f"ground_truth_pddl_plan={self.ground_truth_pddl_plan},\n"
            f"should_supervise_pddl_plan={self.should_supervise_pddl_plan}\n"
            f"proposed_pddl_goals = {self.proposed_pddl_goals}\n"
            f"proposed_pddl_plans = {self.proposed_pddl_plans}\n)"
        )


def load_pddl_supervision(supervision_name: str, verbose: bool = False) -> Dict[str, str]:
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
                pddl_supervision[nl_goal["domain_file"]]["object_list"] = nl_goal["object_list"]
                with open(nl_goal["domain_file"]) as j:
                    pddl_supervision[nl_goal["domain_file"]]["domain"] = OtherDomain(j.read())
                with open(pddl_supervision[nl_goal["domain_file"]]["file_name"]) as g:
                    pddl_supervision[nl_goal["domain_file"]]["pddl_problem_string"] = g.read()

    for domain_file in list(pddl_supervision.keys()):
        if "NL_goal" not in pddl_supervision[domain_file]:
            del pddl_supervision[domain_file]
    if verbose:
        print('Loaded PDDL supervision')
        print('=' * 80)
        print("Loaded additional PDDL supervision from the following domain files:")
        for domain_file in pddl_supervision:
            print(' ', domain_file)
        print('')

    return pddl_supervision


# Planning domains available for the --pddl_domain_name flag.
PLANNING_PDDL_DOMAINS_REGISTRY = dict()


def register_planning_pddl_domain(name):
    def wrapper(f):
        PLANNING_PDDL_DOMAINS_REGISTRY[name] = f
        return f

    return wrapper


def load_pddl_domain(pddl_domain_name: str, initial_pddl_operators: Sequence[str], verbose: bool = False) -> Domain:
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
        print("Initializing with operators: ")
        for o in list(pddl_domain.operators.keys()):
            print(' ', o)
    print('')
    return pddl_domain


# Planning problem datasets available for the --dataset_name flag.
PLANNING_PROBLEMS_REGISTRY = dict()


def register_planning_domain_problems(name):
    def wrapper(f):
        PLANNING_PROBLEMS_REGISTRY[name] = f
        return f

    return wrapper


def get_problem_ids_with_ground_truth_operators(initial_pddl_operators, planning_dataset, split="train"):
    """
    :ret: list of problem IDs where the ground truth plans contain these operators.
    """
    problem_ids = []
    initial_pddl_operators = set(initial_pddl_operators)
    for problem_id in planning_dataset[split]:
        problem = planning_dataset[split][problem_id]
        ground_truth_operators = set(
            [operator[PDDLPlan.PDDL_ACTION] for operator in problem.ground_truth_pddl_plan.plan]
        )
        # Check that this plan doesn't contain any more operators than the initial ones.
        if len(initial_pddl_operators.union(ground_truth_operators)) <= len(initial_pddl_operators):
            problem_ids.append(problem_id)
    return problem_ids


def get_problem_ids_with_initial_plans_prefix(initial_plans_prefix, planning_dataset, split="train"):
    """
    :ret: list of problem IDs where the ground truth plans contain these operators.
    """
    problem_ids = []
    for problem_id in planning_dataset[split]:
        problem = planning_dataset[split][problem_id]
        # This was an outdated method to ensure that we kept in slicing problems.
        if problem.goal_prefix.split("_slice")[0] in initial_plans_prefix:
            problem_ids.append(problem_id)
    return problem_ids


def build_problem_prefix_to_problem_ids(planning_dataset, initial_goal_supervision_prefix, split="train"):
    """
    :ret: list of problem IDs grouped by their operator type.
    """
    problem_prefix_to_problem_ids = defaultdict(list)
    for problem_id in planning_dataset[split]:
        problem = planning_dataset[split][problem_id]
        problem_prefix_to_problem_ids[problem.goal_prefix].append(problem_id)
    return problem_prefix_to_problem_ids


def mark_goal_supervision_problems(planning_dataset, initial_goal_supervision_prefix, goal_supervision_fraction):
    if initial_goal_supervision_prefix == ["SKIP"]:
        return
    problem_prefix_to_problem_ids = build_problem_prefix_to_problem_ids(
        planning_dataset, initial_goal_supervision_prefix, split="train"
    )
    if initial_goal_supervision_prefix == ["ALL"]:
        initial_goal_supervision_prefix = list(problem_prefix_to_problem_ids.keys())

    print("Sampling problems for goal supervision: ")
    total_goal_supervision = 0
    for goal_supervision_type in initial_goal_supervision_prefix:
        num_problems_to_supervise = max(
            1, int(goal_supervision_fraction * len(problem_prefix_to_problem_ids[goal_supervision_type]))
        )
        problems_to_supervise = random.sample(
            problem_prefix_to_problem_ids[goal_supervision_type], num_problems_to_supervise
        )
        for problem_id in problems_to_supervise:
            planning_dataset["train"][problem_id].should_supervise_pddl_goal = True
        print(f"\t {goal_supervision_type} : {num_problems_to_supervise}")
        total_goal_supervision += num_problems_to_supervise
    print(f"Total goal supervision problems: {total_goal_supervision}")


def load_planning_problems_dataset(
    dataset_name,
    dataset_pddl_directory,
    dataset_fraction,
    initial_goal_supervision_fraction,
    initial_goal_supervision_prefix,
    initial_plan_supervision_fraction,
    initial_plan_supervision_prefix,
    initial_pddl_operators,
    domain=None,
    verbose=False,
):
    planning_problem_loader = PLANNING_PROBLEMS_REGISTRY[dataset_name]
    planning_problem_loader.domain = domain
    # Load some fraction of the dataset.

    planning_dataset = planning_problem_loader(
        dataset_pddl_directory=dataset_pddl_directory,
        dataset_fraction=dataset_fraction,
        verbose=verbose,
    )

    print(f"Loaded initial dataset: {dataset_name}")
    print('=' * 80)
    print(f"Initial train problems: {len(planning_dataset['train'])}")

    print(f'Marking problems for goal supervision: fraction={initial_goal_supervision_fraction}, prefix={initial_goal_supervision_prefix}')
    mark_goal_supervision_problems(planning_dataset, initial_goal_supervision_prefix, initial_goal_supervision_fraction)

    print(f'Marking problems for plan supervision: fraction={initial_plan_supervision_fraction}, prefix={initial_plan_supervision_prefix}')
    print('This is not implemented yet. (!!!)')
    print('')

    return planning_dataset
