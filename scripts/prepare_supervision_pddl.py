"""
prepare_supervision_pddl.py | Author: zyzzyva@mit.edu
This preparation script creates the supervision dataset that we use to initialize Codex with example plans.
"""
import argparse
import json
import os
import sys
from pddl import PDDLProblem
import task_planner
from pathlib import Path

import random

random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--supervision_path",
    type=str,
    default="data/domains/supervision_domains",
    help="Location of the original PDDL dataset files.",
)
parser.add_argument(
    "--output_JSON",
    type=str,
    default="data/dataset/supervision-NLgoals-operators.json",
    help="Output to produce the supervision.",
)



def load_domains_and_problems(args):

    files = [f for f in os.listdir(args.supervision_path) if ".pddl" in f]
    domains_and_problems = {domain: [] for domain in files if "domain" in domain}
    for domain in domains_and_problems:
        domain_name = domain.split("_")[0]
        for f in files:
            if domain_name in f and "problem" in f:
                domains_and_problems[domain].append(f)
    return domains_and_problems


def get_operator_sequence(plan):
    operator_sequence = []
    for step in plan:
        operator_sequence.append({"action": step.split()[0], "args": step.split()[1:]})
    return operator_sequence


def get_NL_goals(args):
    f = open(os.path.join(args.supervision_path,"supervision-NL.json"))
    supervision_NL = json.load(f)
    problem_to_NLgoal = {}
    for problem in supervision_NL:
        problem_to_NLgoal[problem["file_name"]] = problem["NL_goal"]
    return problem_to_NLgoal

def plan_domains_and_problems(args, domains_and_problems):
    domains_and_solutions = []
    problem_to_NLgoal = get_NL_goals(args)
    for domain in domains_and_problems:
        for problem in domains_and_problems[domain]:
            solved, plan = task_planner.fd_plan_from_file(
                os.path.join(args.supervision_path, domain),
                os.path.join(args.supervision_path, problem),
            )
            print(solved)
            print(plan)
            with open(os.path.join(args.supervision_path, problem)) as f:
                problem_string = f.read()
            goal_pddl = PDDLProblem(problem_string).ground_truth_goal
            file_name = os.path.join(args.supervision_path, problem)
            goal = problem_to_NLgoal[file_name]
            objects = PDDLProblem(problem_string).parse_problem_objects_pddl()
            if solved:
                operator_sequence = get_operator_sequence(plan)
                domains_and_solutions.append(
                    {
                        "goal_pddl": goal_pddl,
                        "goal": goal,
                        "plan": "",
                        "operator_sequence": operator_sequence,
                        "file_name": os.path.join(args.supervision_path, problem),
                        "domain_file": os.path.join(args.supervision_path, domain),
                        "objects": objects
                    }
                )
    return domains_and_solutions


def main():
    args = parser.parse_args()
    domains_and_problems = load_domains_and_problems(args)
    domains_and_solutions = plan_domains_and_problems(args, domains_and_problems)
    with open(args.output_JSON, "w") as f:
        json.dump(domains_and_solutions, f)


if __name__ == "__main__":
    main()
