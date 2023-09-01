# import llm_operators.codex as codex
from llm_operators.datasets import load_planning_problems_dataset, load_pddl_domain
from llm_operators.codex.goal import propose_goals_for_problems
import pandas as pd
import os
import json

###############################################################
# Here we write basic analysis functions to gain some insight #
# about how well codex is performing                          #
###############################################################


path = "alfred_data/"  # where to save csv files
dataset_name = "alfred_linearized_100"
dataset_fraction = 1
dataset_pddl_directory = "/Users/noakorneev/Documents/llm-operators/data/dataset/alfred_linearized_pddl"
pddl_domain_name = "alfred"
verbose = False
initial_pddl_operators = "GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject"

# Load dataset.
def load_problems_dataset(dataset_name,dataset_fraction,dataset_pddl_directory,initial_operators):
    load_problems = load_planning_problems_dataset(
        dataset_name=dataset_name,
        dataset_fraction=dataset_fraction,
        dataset_pddl_directory=dataset_pddl_directory,
        training_plans_fraction=1,
        initial_pddl_operators=initial_operators
    )
    problems = list(load_problems["train"].values())
    return problems


def divide_solved_unsolved(problems,fraction):
    """
    problems - list of problen objects
    fraction - and integer, where len(problems)/fraction is the size of the solved problems
    returns 2 lists, sovlved and unsolved problems
    """
    n = len(problems)
    solved_problems = problems[:n//fraction]
    unsolved_problems = problems[n//fraction:]

    return solved_problems,unsolved_problems


def collect_ground_truth_and_proposed_goals(solved_problems,unsolved_problems):
    """
    runs codex.propose_goals_for_problems
    returns list of lists, where each is [problem_id,ground_truth, proposed_goal]
    """
    # Load the PDDL domain definition.
    pddl_domain = load_pddl_domain(
        pddl_domain_name, initial_pddl_operators,verbose = False)

    ground_truth_vs_proposed_goals = []

    propose_goals_for_problems(unsolved_problems, solved_problems, pddl_domain)

    for problem in unsolved_problems:
        problem_id = problem.problem_id
        ground_truth = problem.ground_truth_pddl_problem.ground_truth_goal
        proposed_goal = problem.proposed_pddl_goals[0]
        ground_truth_vs_proposed_goals.append([problem_id,ground_truth, proposed_goal])
    return ground_truth_vs_proposed_goals


def csv_gt_vs_proposed_goals(csv_path,csv_name):
    """
    generates a csv comparing porposed and ground truth goals. save it at csv_path/csv_name
    """
    problems = load_problems_dataset(dataset_name,dataset_fraction,dataset_pddl_directory)
    solved_problems,unsolved_problems = divide_solved_unsolved(problems,2)
    ground_truth_vs_proposed_goals = collect_ground_truth_and_proposed_goals(solved_problems,unsolved_problems)
    my_df = pd.DataFrame(ground_truth_vs_proposed_goals)
    my_df.to_csv(os.path.join(csv_path,csv_name), index=False, header=False)

def csv_gt_vs_proposed_goals_from_json(json_file,dataset_pddl_directory,initial_pddl_operators):
    """
    generates csv comparing the proposed goals in json_file to ground truth
    """
    problems = load_problems_dataset(dataset_name,dataset_fraction,dataset_pddl_directory,initial_pddl_operators)
    f = open(json_file)
    experiment_results = json.load(f)
    ground_truth_vs_proposed_goals = [["problem","nl_goal","gt_pddl_goal","proposed_goal"]]

    for problem in problems:
        problem_id = problem.problem_id
        if problem_id in experiment_results:
            nl_goal = problem.language
            ground_truth = problem.ground_truth_pddl_problem.ground_truth_goal
            proposed_goal = experiment_results[problem_id]["codex_output"][0]
            header = "(:goal"
            header_index = proposed_goal.index(header)
            proposed_goal = proposed_goal[header_index:]
            ground_truth_vs_proposed_goals.append([problem_id,nl_goal,ground_truth, proposed_goal])
    my_df = pd.DataFrame(ground_truth_vs_proposed_goals)
    csv_name = json_file[:-5] + ".csv"
    my_df.to_csv(csv_name, index=False, header=False)


json_file = "generated/alfred_linearized_100_supervision_pddl_pick_place_clean/0/alfred_linearized_100_supervision_pddl_pick_place_clean_codex_goals_.json"
csv_gt_vs_proposed_goals_from_json(json_file,dataset_pddl_directory,initial_pddl_operators)