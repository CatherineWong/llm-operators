"""
prototype_main.py | LLM-operators.

Uses LLMs to infer planning operators.
"""
from collections import Counter, defaultdict
import os
import json
import random
import argparse
from datasets import *
from pddl import *
from task_planner import *
from codex import *


DEFAULT_DATASET = "jiahai_20"

DOMAINS_PREFIX = os.path.join(os.getcwd(), "domains")
GENERATED_PREFIX = os.path.join(os.getcwd(), "generated")
PROBLEMS_PREFIX = os.path.join(os.getcwd(), "problems")
PLANS_PREFIX = os.path.join(os.getcwd(), "plans")

MAX_ITERATIONS = 5
EVAL_EVERY = 1
DEFAULT_NUM_TRAIN_OPERATORS = 4
DEFAULT_MAX_GOALS_TO_TRY = 10
DEFAULT_MAX_TRAIN_PLANS_PROMPT = 10
DEFAULT_CODEX_PLAN_SAMPLES = 5
DEFAULT_CODEX_OPERATOR_SAMPLES = 5
DEFAULT_OPERATOR_SYNTHESIS_MAX_ATTEMPTS = 1


def load_domain(dataset):
    # Loads a PDDL domain.
    with open(os.path.join(DOMAINS_PREFIX, dataset + ".pddl")) as f:
        raw_pddl = f.read().lower()
    domain = Domain(pddl_domain=raw_pddl)
    return domain


def load_problems(dataset):
    # Loads PDDL problems and NL descriptions of the goals.
    with open(os.path.join(PROBLEMS_PREFIX, dataset + ".json")) as f:
        raw_problems = json.load(f)
    problem_ids, problems = [], {}
    for problem in raw_problems:
        problem_id = problem["goal_id"]
        problem_ids.append(problem_ids)
        goal_language = problem["goal_language"]
        pddl_problem = problem["pddl_problem"]
        problems[problem_id] = Problem(
            pddl_problem=pddl_problem, goal_language=goal_language
        )
    return problem_ids, problems


def create_train_domain_and_train_plans(dataset, gt_domain, problems):
    # Creates a training domain with a subset of the GT operators and a subset of training supervision using those operators.
    train_plans, gt_plans = {}, {}
    with open(os.path.join(PLANS_PREFIX, dataset + ".json")) as f:
        gt_plans = json.load(f)

    # Plan for any goals that don't already exist.
    gt_plans = maybe_solve_gt_plans(dataset, gt_plans, gt_domain, problems)

    # Create a domain with an ablated set of operators.
    train_domain, train_plans = ablate_gt_domain_and_plans(gt_domain, gt_plans)

    print(
        f"Loaded {len(train_plans)} train plans / {len(gt_plans)} ground truth plans."
    )
    print(
        f"Loaded {len(train_domain.operators)} / {len(gt_domain.operators)} ground truth operators."
    )
    print(f"Starting with the following operators: {train_domain.operators.keys()}")
    return train_plans, train_domain, gt_plans


def maybe_solve_gt_plans(dataset, gt_plans, gt_domain, problems):
    # Solves any goals in the GT operator domain if they don't already exist.
    no_gt_problems = {
        problem_id: problems[problem_id]
        for problem_id in problems.keys()
        if problem_id not in gt_plans
    }
    if len(no_gt_problems) > 0:
        solved_gt_plans = attempt_goals_pddl(
            gt_domain, no_gt_problems, assert_success=True
        )
        gt_plans.update(solved_gt_plans)
        with open(os.path.join(PLANS_PREFIX, dataset + ".json"), "w") as f:
            json.dump(gt_plans, f)
    return gt_plans


def ablate_gt_domain_and_plans(
    gt_domain, gt_plans, num_train_operators=DEFAULT_NUM_TRAIN_OPERATORS
):
    # Constructs the training domain and plans by ablating operators.
    gt_operators_to_problems = defaultdict(list)
    for operator in gt_domain.operators.keys():
        for problem_id, gt_plan in gt_plans.items():
            if operator in " ".join(gt_plan):
                gt_operators_to_problems[operator].append(problem_id)

    # Heuristic: remove all but the most common.
    operators_by_usage = sorted(
        gt_operators_to_problems.keys(), key=lambda o: -len(gt_operators_to_problems[o])
    )
    train_operators = operators_by_usage[:num_train_operators]

    # Operators that need to be learned. This may not be the full set.
    ablated_operators = [o for o in operators_by_usage if o not in train_operators]

    ablated_plans = set().union(
        *[gt_operators_to_problems[o] for o in ablated_operators]
    )
    train_plans = {
        plan_id: gt_plans[plan_id]
        for plan_id in gt_plans
        if plan_id not in ablated_plans
    }
    assert len(train_plans) > 0
    train_domain = Domain(parent_domain=gt_domain)
    train_domain.operators = {
        name: train_domain.operators[name]
        for name in gt_domain.operators
        if name in train_operators
    }
    return train_domain, train_plans


def attempt_goals_pddl(domain, problems, assert_success=False):
    # Attempts to solve PDDL goals given a domain definition using a PDDL solver.
    solved_plans = {}
    for idx, problem_id in enumerate(problems):
        if idx % 10 == 0:
            print(f"Planning now on {idx} / {len(problems)}")
        success, plan = attempt_domain(
            domain.to_string(), problems[problem_id].to_string()
        )
        if assert_success:
            assert success
        if success:
            solved_plans[problem_id] = plan
    return solved_plans


def get_unsolved_problems_to_attempt_codex(
    problems, solved_plans_low_level, max_problems
):
    # Get a batch of goals to attempt with codex.
    all_unsolved_problem_ids = [
        problem_id
        for problem_id in problems
        if problem_id not in solved_plans_low_level
    ]

    # TODO: right now this is just the N shortest unsolved problems.
    problems_by_goal_language = sorted(
        all_unsolved_problem_ids,
        key=lambda p_id: len(problems[p_id].goal_language.split()),
    )
    problem_ids_to_attempt = problems_by_goal_language[:max_problems]
    return problem_ids_to_attempt


def get_proposed_plans_codex(
    unsolved_problem_ids_to_attempt_codex, problems, train_plans, train_domain
):
    proposed_plans = {}
    for problem_id in unsolved_problem_ids_to_attempt_codex:
        # Construct planning prompt.
        prompt = create_plans_prompt(
            problems[problem_id], problems, train_plans, train_domain
        )
        try:
            completions = get_completions(
                prompt,
                temperature=0.1,
                stop="<END>",
                n_samples=DEFAULT_CODEX_PLAN_SAMPLES,
            )
            plans = []
            for c in completions:
                try:
                    plan = eval(c)
                    if type(plan) == list:
                        plans.append(plan)
                except:
                    continue
        except:
            continue
        proposed_plans[problem_id] = plans
    print(f"Got proposed plans for {len(proposed_plans)} problems.")
    return proposed_plans


def create_plans_prompt(problem, problems, train_plans, train_domain):
    prompt = (
        "; Generate PDDL plans for goals specified in natural language.\n"
        + "; Goals are specified with GOAL: <natural language goal>.\n"
        + "; Plans are specified with PLAN: <sequence of PDDL operators>.\n"
    )
    # Sample some training plans.
    train_plan_ids = random.sample(train_plans.keys(), DEFAULT_MAX_TRAIN_PLANS_PROMPT)
    for train_plan_id in train_plan_ids:
        prompt += f"; GOAL: {problems[train_plan_id].goal_language}\n"
        prompt += f"; PLAN: {train_plans[train_plan_id]}\n"

    prompt += f"; GOAL: {problem.goal_language}\n"
    prompt += f"; PLAN: "
    return prompt


def rank_best_proposed_operator_names_from_plan_sketches(
    train_domain, proposed_plan_sketches
):
    # Extracts and ranks best proposed operator names from their plan sketches.
    # Heuristic: orders them by frequency proposed.
    operator_name_counter = Counter()
    operator_names_to_problems = defaultdict(list)
    for proposed_plan_id in proposed_plan_sketches:
        for proposed_plan in proposed_plan_sketches[proposed_plan_id]:
            proposed_operators = [expression.split()[0] for expression in proposed_plan]
            operator_name_counter.update(proposed_operators)
            for operator in proposed_operators:
                operator_names_to_problems[operator].append(proposed_plan_id)
    ranked_operator_names = [o[0] for o in operator_name_counter.most_common()]
    ranked_operator_names = [
        o for o in ranked_operator_names if o not in train_domain.operators.keys()
    ]
    proposed_operators_and_problems = [
        (o, set(operator_names_to_problems[o])) for o in ranked_operator_names
    ]
    return proposed_operators_and_problems


def get_proposed_operator_definitions_codex(
    operator_name, train_domain, max_operator_samples
):
    prompt, separator = create_operator_definitions_prompt(operator_name, train_domain)
    completions = get_completions(
        prompt, temperature=0.1, stop=separator, n_samples=max_operator_samples
    )
    operator_prefix = f"(:action {operator_name}"
    return [operator_prefix + o for o in completions]


def create_operator_definitions_prompt(operator_name, train_domain):
    separator = "\n;;\n"
    prompt = f"{train_domain.predicates}\n;;\n{train_domain.operators_to_string(separator=separator)}"
    prompt += separator + f"(:action {operator_name}"
    return prompt, separator


def verify_proposed_operator(
    operator_name, operator, train_domain, problems_to_verify_on, problems
):
    train_domain.add_operator(operator_name, operator)
    try:
        attempt_goals_pddl(
            train_domain,
            {p_id: problems[p_id] for p_id in problems_to_verify_on},
            assert_success=True,
        )
        return True
    except:
        train_domain.remove_operator(operator_name)
        return False


def synthesize_proposed_operator_definitions_codex(
    operator_name,
    train_domain,
    problems_to_verify_on,
    problems,
    max_operator_samples,
    max_attempts,
):
    # Propose definitions from codex.
    proposed_operator_definitions = get_proposed_operator_definitions_codex(
        operator_name, train_domain, max_operator_samples
    )
    # Keep those that verify on the plans they were proposed for.
    for operator in proposed_operator_definitions:
        try:
            verified = verify_proposed_operator(
                operator_name, operator, train_domain, problems_to_verify_on, problems,
            )
            if verified:
                return operator
        except:
            continue
    print(f"Unable to synthesize operator definition for: {operator_name}")
    return None


def update_train_domain_with_operators_codex(
    gt_domain,
    train_domain,
    unsolved_problem_ids_to_attempt_codex,
    problems,
    proposed_plan_sketches,
):
    # Try updating the train domain with any operators and keep those that work.

    proposed_operator_names_and_problems = rank_best_proposed_operator_names_from_plan_sketches(
        train_domain, proposed_plan_sketches
    )
    print(f"Found {len(proposed_operator_names_and_problems)} proposed operators.")
    for operator_name, problems_to_verify_on in proposed_operator_names_and_problems:
        print(f"Attempting to synthesize an operator for name: {operator_name}")
        operator_definition = synthesize_proposed_operator_definitions_codex(
            operator_name,
            train_domain,
            problems_to_verify_on,
            problems,
            max_operator_samples=DEFAULT_CODEX_OPERATOR_SAMPLES,
            max_attempts=DEFAULT_OPERATOR_SYNTHESIS_MAX_ATTEMPTS,
        )
        if operator_definition is not None:
            # Update the train domain
            print(f"Updating train domain with working operator for: {operator_name}")
            train_domain.add_operator(operator_name, operator_definition)
    return train_domain


def report(
    curr_iteration,
    train_domain,
    gt_domain,
    solved_plans_pddl,
    solved_plans_low_level,
    gt_plans,
    problems,
    dataset,
):
    print("=============")
    print(f"iter: {curr_iteration}")
    print(f"high_level_solved_problems: {len(solved_plans_pddl)} / {len(problems)}")
    print(f"low_level_solved_problems: {len(solved_plans_low_level)} / {len(problems)}")
    print(f"current_operators: {len(train_domain.operators)}")
    print(f"current_operator_names: {train_domain.operators.keys()}")
    # Write out the current plans.
    plan_filename = save_gt_and_learned_plans(
        curr_iteration,
        GENERATED_PREFIX,
        dataset,
        gt_plans,
        solved_plans_pddl,
        problems,
    )
    print(f"saved_plans to: {plan_filename}")
    # Write out the current operator set.
    operator_filename = save_learned_operators(
        curr_iteration, GENERATED_PREFIX, dataset, train_domain, gt_domain
    )
    print("=============")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)


def main():
    args = parser.parse_args()
    gt_domain = load_domain(dataset=args.dataset)
    (problem_ids, problems) = load_problems(dataset=args.dataset)

    train_plans, train_domain, gt_plans = create_train_domain_and_train_plans(
        args.dataset, gt_domain, problems
    )

    for curr_iteration in range(MAX_ITERATIONS):
        if curr_iteration % EVAL_EVERY == 0:
            # Try to solve all of the goals in PDDL
            solved_plans_pddl = attempt_goals_pddl(train_domain, problems)
            # TODO: try to solve goals low-level.
            solved_plans_low_level = solved_plans_pddl

            # Get a subset of the goals to try to solve with codex.
            unsolved_problem_ids_to_attempt_codex = get_unsolved_problems_to_attempt_codex(
                problems, solved_plans_low_level, max_problems=DEFAULT_MAX_GOALS_TO_TRY
            )
        # Propose high level plans that may involve new operators.
        proposed_plan_sketches = get_proposed_plans_codex(
            unsolved_problem_ids_to_attempt_codex, problems, train_plans, train_domain
        )
        # Update the domain definition with new operator definitions.
        train_domain = update_train_domain_with_operators_codex(
            gt_domain,
            train_domain,
            unsolved_problem_ids_to_attempt_codex,
            problems,
            proposed_plan_sketches=proposed_plan_sketches,
        )
        report(
            curr_iteration,
            train_domain,
            gt_domain,
            solved_plans_pddl,
            solved_plans_low_level,
            gt_plans,
            problems,
            dataset=args.dataset,
        )


if __name__ == "__main__":
    main()
