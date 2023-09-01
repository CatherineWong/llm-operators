"""
test_codex.py | Tests for codex.py.
"""

import os
import copy

import llm_operators.codex as codex
from llm_operators.pddl import Domain, PDDLProblem
from llm_operators.datasets import Problem
from llm_operators.datasets.alfred import load_alfred_pddl_file


def create_mock_domain(ALFRED_DOMAIN_FILE_PATH="data/domains/alfred.pddl"):
    # Create a structured test object from a planning domain.
    # Loads a PDDL domain.
    with open(os.path.join(ALFRED_DOMAIN_FILE_PATH)) as f:
        raw_pddl = f.read()
    domain = Domain(pddl_domain=raw_pddl)
    domain.ground_truth_operators = {
        o: copy.deepcopy(domain.operators[o]) for o in domain.operators}
    return domain


def create_mock_ablated_domain(domain, operators_to_keep=("GotoLocation", "OpenObject")):
    # Create a mock domain with only a few operators.
    for o in list(domain.operators.keys()):
        if o not in operators_to_keep:
            domain.remove_operator(o)
    return domain


def create_mock_operator_uses():
    # Create a few sample operator uses for the ALFRED domain.
    return {
        "GotoLocation": [
            "(GotoLocation agent1 loc_1 loc_2)",
            "(GotoLocation agent1 loc_2 loc_3)",
            "(GotoLocation agent1 loc_2 loc_5)",
        ],
        "OpenObject": [
            "(OpenObject agent1 loc_1 Microwave)",
            "(OpenObject agent1 Fridge)",
            "(OpenObject agent1 GarbageCan)",
        ],
        "CloseObject": [
            "(CloseObject agent1 loc_1 Microwave)",
            "(CloseObject agent1 loc_2 Fridge)",
            "(CloseObject agent1 loc_3 Fridge)",
        ],
    }


def create_mock_unsolved_problem_list(dataset_pddl_directory = "data/dataset/alfred_pddl/"):
    # Create a list of unsolved Problem objects
    problem_json = [{'goal': 'put a slice of vegetable on a counter.', 'operator_sequence': [{'action': 'GotoLocation', 'args': ['diningtable']}, {'action': 'PickupObject', 'args': ['knife']}, {'action': 'SliceObject', 'args': ['lettuce']}, {'action': 'GotoLocation', 'args': ['fridge']}, {'action': 'PutObject', 'args': ['knife', 'fridge']}, {'action': 'GotoLocation', 'args': ['diningtable']}, {'action': 'PickupObject', 'args': ['lettuce']}, {'action': 'GotoLocation', 'args': ['fridge']}, {'action': 'CoolObject', 'args': ['lettuce']}, {'action': 'GotoLocation', 'args': ['diningtable']}, {'action': 'PutObject', 'args': ['lettuce', 'diningtable']}], 'file_name': 'train/pick_cool_then_place_in_recep-LettuceSliced-None-DiningTable-17/trial_T20190909_070538_437648'},
                     {'goal': 'put two candles in a cabinet underneath the sink.', 'operator_sequence': [{'action': 'GotoLocation', 'args': ['countertop']}, {'action': 'PickupObject', 'args': ['candle']}, {'action': 'GotoLocation', 'args': ['cabinet']}, {'action': 'PutObject', 'args': ['candle', 'cabinet']}, {'action': 'GotoLocation', 'args': ['toilet']}, {'action': 'PickupObject', 'args': ['candle']}, {'action': 'GotoLocation', 'args': ['cabinet']}, {'action': 'PutObject', 'args': ['candle', 'cabinet']}], 'file_name': 'train/pick_two_obj_and_place-Candle-None-Cabinet-414/trial_T20190908_190650_163902'}]
    problem_ids = ['0_put a slice of vegetable on a counter.',
                   '3_put two candles in a cabinet underneath the sink.']
    languages = ['put a slice of vegetable on a counter.',
                 'put two candles in a cabinet underneath the sink.']
    ground_truth_pddl_plans = [[{'action': 'GotoLocation', 'args': ['diningtable']}, {'action': 'PickupObject', 'args': ['knife']}, {'action': 'SliceObject', 'args': ['lettuce']}, {'action': 'GotoLocation', 'args': ['fridge']}, {'action': 'PutObject', 'args': ['knife', 'fridge']}, {'action': 'GotoLocation', 'args': ['diningtable']}, {'action': 'PickupObject', 'args': ['lettuce']}, {'action': 'GotoLocation', 'args': ['fridge']}, {'action': 'CoolObject', 'args': ['lettuce']}, {'action': 'GotoLocation', 'args': ['diningtable']}, {'action': 'PutObject', 'args': ['lettuce', 'diningtable']}],
                               [{'action': 'GotoLocation', 'args': ['countertop']}, {'action': 'PickupObject', 'args': ['candle']}, {'action': 'GotoLocation', 'args': ['cabinet']}, {'action': 'PutObject', 'args': ['candle', 'cabinet']}, {'action': 'GotoLocation', 'args': ['toilet']}, {'action': 'PickupObject', 'args': ['candle']}, {'action': 'GotoLocation', 'args': ['cabinet']}, {'action': 'PutObject', 'args': ['candle', 'cabinet']}]]
    ground_truth_pddl_problems = [PDDLProblem(
                ground_truth_pddl_problem_string=load_alfred_pddl_file(
                    dataset_pddl_directory, problem_json[j]["file_name"]))
                for j in range(len(problem_json))]

    return [Problem(
        problem_id=problem_ids[i],
        language=languages[i],
        ground_truth_pddl_plan=ground_truth_pddl_plans[i],
        ground_truth_pddl_problem=ground_truth_pddl_problems[i],
        should_supervise_pddl_plan=True
        )
        for i in range(len(problem_ids))]


def create_mock_solved_problem_list(dataset_pddl_directory = "data/dataset/alfred_pddl/"):
    # Create a list of solved Problem objects

    problem_json = [{'goal': 'put the cooked egg in the kitchen sink.', 'operator_sequence': [{'action': 'GotoLocation', 'args': ['countertop']}, {'action': 'PickupObject', 'args': ['egg']}, {'action': 'GotoLocation', 'args': ['microwave']}, {'action': 'HeatObject', 'args': ['egg']}, {'action': 'GotoLocation', 'args': ['sinkbasin']}, {'action': 'PutObject', 'args': ['egg', 'sinkbasin']}], 'file_name': 'train/pick_heat_then_place_in_recep-Egg-None-SinkBasin-20/trial_T20190908_205050_000947'},
                    {'goal': 'wash a teapot to put it away under the counter.', 'operator_sequence': [{'action': 'GotoLocation', 'args': ['stoveburner']}, {'action': 'PickupObject', 'args': ['kettle']}, {'action': 'GotoLocation', 'args': ['sinkbasin']}, {'action': 'CleanObject', 'args': ['kettle']}, {'action': 'GotoLocation', 'args': ['cabinet']}, {'action': 'PutObject', 'args': ['kettle', 'cabinet']}], 'file_name': 'train/pick_clean_then_place_in_recep-Kettle-None-Cabinet-2/trial_T20190909_043103_418752'}]
    problem_ids = ['6_put the cooked egg in the kitchen sink.',
                   '13_wash a teapot to put it away under the counter.']
    languages = ['put the cooked egg in the kitchen sink.',
                 'wash a teapot to put it away under the counter.']
    ground_truth_pddl_plans = [[{'action': 'GotoLocation', 'args': ['countertop']}, {'action': 'PickupObject', 'args': ['egg']}, {'action': 'GotoLocation', 'args': ['microwave']}, {'action': 'HeatObject', 'args': ['egg']}, {'action': 'GotoLocation', 'args': ['sinkbasin']}, {'action': 'PutObject', 'args': ['egg', 'sinkbasin']}],
                               [{'action': 'GotoLocation', 'args': ['stoveburner']}, {'action': 'PickupObject', 'args': ['kettle']}, {'action': 'GotoLocation', 'args': ['sinkbasin']}, {'action': 'CleanObject', 'args': ['kettle']}, {'action': 'GotoLocation', 'args': ['cabinet']}, {'action': 'PutObject', 'args': ['kettle', 'cabinet']}]]
    ground_truth_pddl_problems = [PDDLProblem(
        ground_truth_pddl_problem_string=load_alfred_pddl_file(
            dataset_pddl_directory, problem_json[j]["file_name"]))
        for j in range(len(problem_json))]

    return [Problem(
        problem_id=problem_ids[i],
        language=languages[i],
        ground_truth_pddl_plan=ground_truth_pddl_plans[i],
        ground_truth_pddl_problem=ground_truth_pddl_problems[i],
        should_supervise_pddl_plan=True
        )
        for i in range(len(problem_ids))]


def test_propose_operator_definition():
    mock_ground_truth_domain = create_mock_domain()
    ablated_domain = create_mock_ablated_domain(
        domain=create_mock_domain(), operators_to_keep=["GotoLocation", "OpenObject"]
    )
    mock_uses = create_mock_operator_uses()

    operator_name_to_define = "CloseObject"
    operator_definitions = codex.propose_operator_definition(
        current_domain=ablated_domain,
        operator_name_to_define=operator_name_to_define,
        operator_uses=mock_uses,
        verbose=True,
    )
    assert len(operator_definitions) == 1
    # Compare it to the ground truth.
    print("Ground truth: ")
    print(mock_ground_truth_domain.operators[operator_name_to_define])
    print("Proposed: ")
    print(operator_definitions[0])


def test_propose_plans_for_problems():
    unsolved_problems = create_mock_unsolved_problem_list()
    solved_problems = create_mock_solved_problem_list()
    current_domain = create_mock_domain()
    codex.propose_plans_for_problems(unsolved_problems, solved_problems, current_domain)
    for problem in unsolved_problems:
        assert(len(problem.proposed_pddl_plans) != 0)
        print(problem)


def test_propose_PDDL_goals_for_problems():
    unsolved_problems = create_mock_unsolved_problem_list()
    solved_problems = create_mock_solved_problem_list()
    current_domain = create_mock_domain()
    codex.propose_goals_for_problems(unsolved_problems, solved_problems, current_domain,initial_pddl_predicates = ["test"],use_mock = False, output_directory = './generated/mock_data')
    for problem in unsolved_problems:
        assert(len(problem.proposed_pddl_goals) != 0)
        print(problem)
