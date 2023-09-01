"""
prepare_alfred_pddl.py | Author: zyzzyva@mit.edu.

This preparation script was used to create the modified ALFRED PDDL dataset that we use for planning. Modifies goals from: https://github.com/askforalfred/alfred/blob/master/gen/goal_library.py
"""
import argparse
import json
import os
import sys
from llm_operators.pddl import PDDLProblem
import llm_operators.task_planner as task_planner
from pathlib import Path

import random

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pddl_domain_file",
    type=str,
    default="data/domains/alfred_linearized.pddl",
    help="Location of the original PDDL domain file.",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data/dataset/alfred_pddl",
    help="Location of the original PDDL dataset files.",
)
parser.add_argument(
    "--output_dataset_path",
    type=str,
    default="data/dataset/alfred_linearized_pddl",
    help="Location of the original PDDL dataset files.",
)
parser.add_argument(
    "--tasks_json",
    type=str,
    default="data/dataset/alfred-NLgoals-operators.json",
    help="Location of the JSON containing the original problems.",
)
parser.add_argument(
    "--skip_exists", action="store_true", help="If exists, skip it.",
)
parser.add_argument(
    "--use_synthetic_goal", action="store_true", help="Use a synthetic goal.",
)
parser.add_argument(
    "--use_cot", action="store_true", help="Include chain of thought prompting.",
)


def load_alfred_nl_goals(args):
    with open(args.tasks_json) as f:
        return json.load(f)


def get_goals_for_prefix(goal_prefix, goals):
    goals_for_prefix = []
    for goal in goals:
        curr_goal_prefix = goal.split("/")[1].split("-")[0]

        if (
            "slice" not in goal_prefix
            and curr_goal_prefix == goal_prefix
            and not "Sliced" in goal.split("/")[1]
        ):
            goals_for_prefix.append(goal)
        if (
            "slice" in goal_prefix
            and curr_goal_prefix == "_".join(goal_prefix.split("_")[:-1])
            and "Sliced" in goal.split("/")[1]
        ):
            goals_for_prefix.append(goal)
        else:
            pass
    return goals_for_prefix


def preprocess_problem_domain(args, goal_file, goal_data):
    goal_pddl_file = os.path.join(args.dataset_path, goal_file, "problem_0.pddl")
    with open(goal_pddl_file) as f:
        problem_file = f.read()

    with open(args.pddl_domain_file) as f:
        domain_file = f.read()

    # Remove the cost predicate.
    problem_file = "\n".join(
        [l for l in problem_file.split("\n") if "totalCost" not in l]
    )

    # Remove the distance predicate.
    problem_file = "\n".join(
        [l for l in problem_file.split("\n") if "distance" not in l]
    )

    # Remove the canContain predicate.
    problem_file = "\n".join(
        [l for l in problem_file.split("\n") if "canContain" not in l]
    )

    # Dedup files.
    deduped_lines = []
    for l in problem_file.split("\n"):
        if "loc" in l:
            if l not in deduped_lines:
                deduped_lines.append(l)
        else:
            deduped_lines.append(l)
    problem_file = "\n".join(deduped_lines)

    return problem_file, domain_file


def write_preprocessed_file(args, goal_file, problem_string):
    output_pddl_file = os.path.join(args.output_dataset_path, goal_file)
    print(f"Writing to: {os.path.join(output_pddl_file, 'problem_0.pddl')}\n\n")
    Path(output_pddl_file).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_pddl_file, "problem_0.pddl"), "w") as f:
        f.write(problem_string)


def preprocess_file_and_replace_goal(args, goal_file, goal_data, alternate_goal):
    problem_file, domain_file = preprocess_problem_domain(args, goal_file, goal_data)
    new_goal_pddl_file = os.path.join(
        args.output_dataset_path, goal_file, "problem_0.pddl"
    )
    if args.skip_exists and os.path.exists(new_goal_pddl_file):
        return True

    problem = PDDLProblem(ground_truth_pddl_problem_string=problem_file)
    alternate_problem = problem.get_pddl_string_with_proposed_goal(alternate_goal)

    # Check that we can solve the modified version.
    solved, plan = task_planner.fd_plan_from_strings(domain_file, alternate_problem)
    if not solved or len(plan) < 1:
        print(f"Error at: {goal_file}")
        return False
    print("Linearized goal: ")
    print(problem.ground_truth_goal)
    print(f"Successfully solved with: {plan}")
    write_preprocessed_file(args, goal_file, problem_string=alternate_problem)
    return True


def pick_and_place_simple(args, goal_file, goal_data):
    # Reparse the goal.
    curr_goal_prefix = goal_file.split("/")[1]
    goal_obj, goal_recep = (
        curr_goal_prefix.split("-")[1],
        curr_goal_prefix.split("-")[3],
    )
    alternate_goal = f"""
    (:goal
        (exists (?r - receptacle)
        (exists (?o - object)
            (and
                (objectType ?o {goal_obj}Type)
                (receptacleType ?r {goal_recep}Type)
                (inReceptacle ?o ?r)
            )
    )))"""
    return preprocess_file_and_replace_goal(args, goal_file, goal_data, alternate_goal)


def pick_and_place_simple_slice(args, goal_file, goal_data):
    # Reparse the goal.
    curr_goal_prefix = goal_file.split("/")[1]
    goal_obj, goal_recep = (
        curr_goal_prefix.split("-")[1],
        curr_goal_prefix.split("-")[3],
    )
    goal_obj = goal_obj.replace("Sliced", "")
    alternate_goal = f"""
    (:goal
        (exists (?r - receptacle)
        (exists (?o - object)
            (and
                (objectType ?o {goal_obj}Type)
                (receptacleType ?r {goal_recep}Type)
                (inReceptacle ?o ?r)
                (sliceable ?o)
                (isSliced ?o)
            )
    )))"""
    return preprocess_file_and_replace_goal(args, goal_file, goal_data, alternate_goal)


def pick_clean_then_place_in_recep(args, goal_file, goal_data):
    # Reparse the goal.
    curr_goal_prefix = goal_file.split("/")[1]
    goal_obj, goal_recep = (
        curr_goal_prefix.split("-")[1],
        curr_goal_prefix.split("-")[3],
    )
    alternate_goal = f"""
    (:goal
        (exists (?r - receptacle)
        (exists (?o - object)
            (and
                (objectType ?o {goal_obj}Type)
                (receptacleType ?r {goal_recep}Type)
                (inReceptacle ?o ?r)
                (cleanable ?o)
                (isClean ?o)
            )
    )))"""
    return preprocess_file_and_replace_goal(args, goal_file, goal_data, alternate_goal)


def pick_heat_then_place_in_recep_slice(args, goal_file, goal_data):
    # Reparse the goal.
    curr_goal_prefix = goal_file.split("/")[1]
    goal_obj, goal_recep = (
        curr_goal_prefix.split("-")[1],
        curr_goal_prefix.split("-")[3],
    )
    goal_obj = goal_obj.replace("Sliced", "")
    alternate_goal = f"""
    (:goal
        (exists (?r - receptacle)
        (exists (?o - object)
            (and
                (objectType ?o {goal_obj}Type)
                (receptacleType ?r {goal_recep}Type)
                (inReceptacle ?o ?r)
                (heatable ?o)
                (isHot ?o)
                (sliceable ?o)
                (isSliced ?o)
            )
    )))"""
    return preprocess_file_and_replace_goal(args, goal_file, goal_data, alternate_goal)


def pick_heat_then_place_in_recep(args, goal_file, goal_data):
    # Reparse the goal.
    curr_goal_prefix = goal_file.split("/")[1]
    goal_obj, goal_recep = (
        curr_goal_prefix.split("-")[1],
        curr_goal_prefix.split("-")[3],
    )
    alternate_goal = f"""
    (:goal
        (exists (?r - receptacle)
        (exists (?o - object)
            (and
                (objectType ?o {goal_obj}Type)
                (receptacleType ?r {goal_recep}Type)
                (inReceptacle ?o ?r)
                (heatable ?o)
                (isHot ?o)
            )
    )))"""
    return preprocess_file_and_replace_goal(args, goal_file, goal_data, alternate_goal)


def pick_cool_then_place_in_recep(args, goal_file, goal_data):
    # Reparse the goal.
    curr_goal_prefix = goal_file.split("/")[1]
    goal_obj, goal_recep = (
        curr_goal_prefix.split("-")[1],
        curr_goal_prefix.split("-")[3],
    )
    alternate_goal = f"""
    (:goal
        (exists (?r - receptacle)
        (exists (?o - object)
            (and
                (objectType ?o {goal_obj}Type)
                (receptacleType ?r {goal_recep}Type)
                (inReceptacle ?o ?r)
                (coolable ?o)
                (isCool ?o)
            )
    )))"""
    return preprocess_file_and_replace_goal(args, goal_file, goal_data, alternate_goal)


def pick_cool_then_place_in_recep_slice(args, goal_file, goal_data):
    # Reparse the goal.
    curr_goal_prefix = goal_file.split("/")[1]
    goal_obj, goal_recep = (
        curr_goal_prefix.split("-")[1],
        curr_goal_prefix.split("-")[3],
    )
    goal_obj = goal_obj.replace("Sliced", "")
    alternate_goal = f"""
    (:goal
        (exists (?r - receptacle)
        (exists (?o - object)
            (and
                (objectType ?o {goal_obj}Type)
                (receptacleType ?r {goal_recep}Type)
                (inReceptacle ?o ?r)
                (coolable ?o)
                (isCool ?o)
                (sliceable ?o)
                (isSliced ?o)
            )
    )))"""
    return preprocess_file_and_replace_goal(args, goal_file, goal_data, alternate_goal)


# Go to a location, and toggle on the light.
def look_at_obj_in_light(args, goal_file, goal_data):
    # Reparse the goal.
    curr_goal_prefix = goal_file.split("/")[1]
    goal_obj, goal_recep = (
        curr_goal_prefix.split("-")[1],
        curr_goal_prefix.split("-")[3],
    )

    alternate_goal = f"""
    (:goal
        (exists (?a - agent)
        (exists (?r - receptacle)
        (exists (?o - object)
        (exists (?ot - object)
        (exists (?l - location)
            (and
                (objectType ?ot {goal_recep}Type)
                (toggleable ?ot)
                (isToggled ?ot)
                (objectAtLocation ?ot ?l)
                (atLocation ?a ?l)
                (objectType ?o {goal_obj}Type)
                (holds ?a ?o)
            )
    ))))))"""
    return preprocess_file_and_replace_goal(args, goal_file, goal_data, alternate_goal)


GOAL_PREFIXES = {
    "pick_and_place_simple": pick_and_place_simple,
    "pick_clean_then_place_in_recep": pick_clean_then_place_in_recep,
    "pick_heat_then_place_in_recep": pick_heat_then_place_in_recep,
    "pick_cool_then_place_in_recep": pick_cool_then_place_in_recep,
    "pick_and_place_simple_slice": pick_and_place_simple_slice,
    "pick_heat_then_place_in_recep_slice": pick_heat_then_place_in_recep_slice,
    "pick_cool_then_place_in_recep_slice": pick_cool_then_place_in_recep_slice,
    "look_at_obj_in_light": look_at_obj_in_light,
}

def generate_synthetic_goal(goal_file):
    """
    A very basic function to generate a synthetic description from a goal file.
    """
    problem_name = goal_file.split("/")[1]
    problem_type = problem_name.split("-")[0]
    problem_args = problem_name.split("-")[1:]
    print(problem_name)
    print(problem_args)

    if "Sliced" in problem_args[0]:
        problem_args[0] = problem_args[0].replace("Sliced", ", slice it")

    if problem_type == "pick_and_place_simple":
        return "Pick up the " + problem_args[0] + " and place it in the " + problem_args[2] + "."
    elif problem_type == "pick_clean_then_place_in_recep":
        return "Pick up the " + problem_args[0] + " and clean it. Then place it in the " + problem_args[2] + "."
    elif problem_type == "pick_heat_then_place_in_recep":
        return "Pick up the " + problem_args[0] + " and heat it. Then place it in the " + problem_args[2] + "."
    elif problem_type == "pick_cool_then_place_in_recep":
        return "Pick up the " + problem_args[0] + " and cool it. Then place it in the " + problem_args[2] + "."
    elif problem_type == "look_at_obj_in_light":
        return "Look at the " + problem_args[0] + " in the light of the " + problem_args[2] + "."

def generate_cot(goal_file):
    """
    Generate chain-of-thought prompts for goals.
    """
    problem_name = goal_file.split("/")[1]
    problem_type = problem_name.split("-")[0]
    problem_args = problem_name.split("-")[1:]

    thought = problem_name.rsplit("-", 1)[0]

    """thought = ";; Two Relevant Objects: 1) " + problem_args[0] + " and 2) " + problem_args[2] + ". "
    thought += ";; Problem Type: " + problem_type + "."
    thought += ";; Simplified Goal: "

    sliced = (True if "Sliced" in problem_args[0] else False)
    if "Sliced" in problem_args[0]:
        problem_args[0] = problem_args[0].replace("Sliced", "")
        problem_args[0] = "AND SLICE " + problem_args[0]

    if problem_type == "pick_and_place_simple":
        thought += "PICKUP " + problem_args[0] + " and PLACE in " + problem_args[2] + "."
    elif problem_type == "pick_clean_then_place_in_recep":
        thought += "PICKUP and CLEAN " + problem_args[0] + ". Then PLACE in " + problem_args[2] + "."
    elif problem_type == "pick_heat_then_place_in_recep":
        thought += "PICKUP and HEAT " + problem_args[0] + ". Then PLACE in " + problem_args[2] + "."
    elif problem_type == "pick_cool_then_place_in_recep":
        thought += "PICKUP and COOL " + problem_args[0] + ". Then PLACE in " + problem_args[2] + "."
    elif problem_type == "look_at_obj_in_light":
        thought += "PICKUP " + problem_args[0] + " and TOGGLE the light of " + problem_args[2] + "."""

    return thought

def main():
    args = parser.parse_args()
    alfred_nl_goals = load_alfred_nl_goals(args)

    """sucessfully_parsed = {}
    for split in alfred_nl_goals:
        print(f"Preparing split: {split} with {len(alfred_nl_goals[split])} tasks")
        goals = {goal["file_name"]: goal for goal in alfred_nl_goals[split]}
        sucessfully_parsed[split] = []

        for goal_prefix in GOAL_PREFIXES:
            goals_for_prefix = get_goals_for_prefix(goal_prefix, goals)
            print(f"Modifying {len(goals_for_prefix)} goals for {goal_prefix}")
            goal_modification_fn = GOAL_PREFIXES[goal_prefix]
            for idx, goal in enumerate(goals_for_prefix):
                print(
                    f"Now on: {idx} / {len(goals_for_prefix)} for goals of type {goal_prefix}"
                )
                successful = goal_modification_fn(args, goal, goals[goal])
                if successful:
                    sucessfully_parsed[split].append(goal)
    # Log out any unsuccessful.
    not_sucessfully_parsed = {}
    for split in alfred_nl_goals:
        not_sucessfully_parsed[split] = []
        for goal in alfred_nl_goals[split]:
            if goal not in sucessfully_parsed[split]:
                not_sucessfully_parsed[split].append(goal["file_name"])
    with open(os.path.join(args.output_dataset_path, "not_included.json"), "w") as f:
        json.dump(not_sucessfully_parsed, f)"""

    # Take a subset of the problems for a shorter debug set.
    MAX_SET = 250
    dataset_name = f"data/dataset/alfred-cot-{MAX_SET}-NLgoals-operators.json"
    dataset_subset = {}
    for split in alfred_nl_goals:
        dataset_subset[split] = []
        for goal_prefix in GOAL_PREFIXES:
            goals = {goal["file_name"]: goal for goal in alfred_nl_goals[split]}
            goals_for_prefix = get_goals_for_prefix(goal_prefix, goals)
            max_goals = int(MAX_SET / len(GOAL_PREFIXES))
            successfully_parsed_goals = [
                g
                for g in goals_for_prefix
                if os.path.exists(os.path.join(args.output_dataset_path, g))
            ]
            # Randomly select some.
            prefix_subset = random.sample(
                successfully_parsed_goals,
                min(max_goals, len(successfully_parsed_goals)),
            )
            for g in prefix_subset:
                goals[g]["goal_prefix"] = goal_prefix
                if args.use_synthetic_goal:
                    goals[g]["goal"] = generate_synthetic_goal(goals[g]["file_name"])
                if args.use_cot:
                    goals[g]["cot"] = generate_cot(goals[g]["file_name"])
                dataset_subset[split].append(goals[g])
    with open(dataset_name, "w") as f:
        json.dump(dataset_subset, f)


if __name__ == "__main__":
    main()
