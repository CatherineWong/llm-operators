#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import json

import numpy as np

from datasets.core import (
    Problem,
    register_planning_pddl_domain,
    register_planning_domain_problems,
)
from datasets.dataset_utils import load_pddl_file_with_operators
from pddl import PDDLProblem


ALFRED_PDDL_DOMAIN_NAME = "alfred"


@register_planning_pddl_domain(ALFRED_PDDL_DOMAIN_NAME)
def load_alfred_pddl_domain(verbose=False):
    ALFRED_DOMAIN_FILE_PATH = "data/domains/alfred.pddl"
    domain = load_pddl_file_with_operators(
        domain_name=ALFRED_PDDL_DOMAIN_NAME,
        file_path=ALFRED_DOMAIN_FILE_PATH,
        verbose=verbose,
    )
    # Remove the functions from the file and from all of the operators.
    domain.functions = ""

    def remove_functions(operator_body):
        return "\n".join([l for l in operator_body.split("\n") if "totalCost" not in l])

    for operator in domain.ground_truth_operators:
        operator_body = domain.ground_truth_operators[operator]
        domain.ground_truth_operators[operator] = remove_functions(operator_body)
    for operator in domain.operators:
        operator_body = domain.operators[operator]
        domain.operators[operator] = remove_functions(operator_body)
    return domain


ALFWORLD_PDDL_DOMAIN_NAME = "alfworld"


@register_planning_pddl_domain(ALFWORLD_PDDL_DOMAIN_NAME)
def load_alfworld_pddl_domain(verbose=False):
    ALFWORLD_DOMAIN_FILE_PATH = "data/domains/alfworld.pddl"
    return load_pddl_file_with_operators(
        domain_name=ALFWORLD_PDDL_DOMAIN_NAME,
        file_path=ALFWORLD_DOMAIN_FILE_PATH,
        verbose=verbose,
    )


ALFRED_LINEARIZED_PDDL_DOMAIN_NAME = "alfred_linearized"


@register_planning_pddl_domain(ALFRED_LINEARIZED_PDDL_DOMAIN_NAME)
def load_alfred_linearized_pddl_domain(verbose=False):
    ALFRED_LINEARIZED_PDDL_FILE_PATH = "data/domains/alfred_linearized.pddl"
    domain = load_pddl_file_with_operators(
        domain_name=ALFRED_LINEARIZED_PDDL_DOMAIN_NAME,
        file_path=ALFRED_LINEARIZED_PDDL_FILE_PATH,
        verbose=verbose,
    )
    domain.operator_canonicalization = {
        "PickupObjectInReceptacle": "PickupObject",
        "PickupObjectNotInReceptacle": "PickupObject",
        "PutObjectInReceptacle": "PutObject",
        "PutReceptacleObjectInReceptacle": "PutObject",
    }
    domain.add_additional_constants(ALFRED_LINEARIZED_CODEX_TYPES)
    domain.codex_types = ALFRED_LINEARIZED_CODEX_TYPES

    for predicate in ['receptacleType', 'objectType']:
        domain.ground_truth_predicates[predicate].mark_static()
    return domain


# ALFRED Dataset.
ALFRED_DATASET_NAME = "alfred"
ALFRED_DATASET_PATH = "data/dataset/alfred-NLgoals-operators.json"

# Use the linearized problems
ALFRED_DEFAULT_PDDL_DIRECTORY = "data/dataset/alfred_linearized_pddl"


@register_planning_domain_problems(ALFRED_DATASET_NAME)
def load_alfred_planning_domain_problems(
    dataset_pddl_directory=ALFRED_DEFAULT_PDDL_DIRECTORY,
    dataset_fraction=1.0,
    verbose=False,
):
    """
    splits are: train, valid_seen, valid_unseen
    :ret: {
        split: {problem_id : Problem}
        }
    for the ALFRED dataset.
    """
    # Location of the local alfred-NLgoals-operators JSON.
    with open(ALFRED_DATASET_PATH) as f:
        alfred_json = json.load(f)

    dataset = dict()
    for dataset_split in alfred_json:
        dataset[dataset_split] = dict()
        # Get some fraction of the dataset to load.
        num_to_take = int(np.ceil(dataset_fraction * len(alfred_json[dataset_split])))
        fraction_split = random.sample(list(alfred_json[dataset_split]), num_to_take)
        for problem_json in fraction_split:
            problem_id = problem_json["file_name"]
            goal_language = problem_json["goal"]
            ground_truth_pddl_plan = problem_json["operator_sequence"]
            ground_truth_pddl_problem = PDDLProblem(
                ground_truth_pddl_problem_string=load_alfred_pddl_file(
                    dataset_pddl_directory, problem_json["file_name"]
                )
            )
            new_problem = Problem(
                problem_id=problem_id,
                dataset_split=dataset_split,
                language=goal_language,
                ground_truth_pddl_plan=ground_truth_pddl_plan,
                ground_truth_pddl_problem=ground_truth_pddl_problem,
            )
            new_problem.constants_in_problem_file = True
            dataset[dataset_split][problem_id] = new_problem

    if verbose:
        print(
            f"\nload_alfred_planning_domain_problems: loaded {ALFRED_DATASET_NAME} from {ALFRED_DATASET_PATH}"
        )
        for dataset_split in dataset:
            print(
                f"{dataset_split} : {len(dataset[dataset_split])} / original {len(alfred_json[dataset_split])} problems"
            )
    return dataset


# Development subset of 100 learning problems.
ALFRED_LINEARIZED_100_DATASET_NAME = "alfred_linearized_100"
ALFRED_LINEARIZED_100_DATASET_PATH = (
    "data/dataset/alfred-linearized-100-NLgoals-operators.json"
)


@register_planning_domain_problems(ALFRED_LINEARIZED_100_DATASET_NAME)
def load_alfred_linearized_planning_domain_problems(
    dataset_pddl_directory=ALFRED_DEFAULT_PDDL_DIRECTORY,
    dataset_fraction=1.0,
    verbose=False,
):
    """
    splits are: train, valid_seen, valid_unseen
    :ret: {
        split: {problem_id : Problem}
        }
    for the ALFRED dataset.
    """
    # Location of the local alfred-NLgoals-operators JSON.
    with open(ALFRED_LINEARIZED_100_DATASET_PATH) as f:
        alfred_json = json.load(f)

    dataset = dict()
    for dataset_split in alfred_json:
        dataset[dataset_split] = dict()
        # Get some fraction of the dataset to load.
        num_to_take = int(np.ceil(dataset_fraction * len(alfred_json[dataset_split])))
        fraction_split = random.sample(list(alfred_json[dataset_split]), num_to_take)
        for problem_json in fraction_split:
            problem_id = problem_json["file_name"]
            goal_language = problem_json["goal"]
            ground_truth_pddl_plan = problem_json["operator_sequence"]
            goal_prefix = (
                problem_json["goal_prefix"] if "goal_prefix" in problem_json else ""
            )

            ground_truth_pddl_problem = PDDLProblem(
                ground_truth_pddl_problem_string=load_alfred_pddl_file(
                    dataset_pddl_directory, problem_json["file_name"]
                )
            )
            new_problem = Problem(
                problem_id=problem_id,
                dataset_split=dataset_split,
                language=goal_language,
                ground_truth_pddl_plan=ground_truth_pddl_plan,
                ground_truth_pddl_problem=ground_truth_pddl_problem,
                goal_prefix=goal_prefix,
            )
            new_problem.constants_in_problem_file = True
            dataset[dataset_split][problem_id] = new_problem

    if verbose:
        print(
            f"\nload_alfred_linearized_planning_domain_problems: loaded {ALFRED_DATASET_NAME} from {ALFRED_LINEARIZED_100_DATASET_PATH}"
        )
        for dataset_split in dataset:
            print(
                f"{dataset_split} : {len(dataset[dataset_split])} / original {len(alfred_json[dataset_split])} problems"
            )
    return dataset


def load_alfred_pddl_file(
    dataset_pddl_directory, problem_directory, pddl_file="problem_0.pddl"
):
    with open(os.path.join(dataset_pddl_directory, problem_directory, pddl_file)) as f:
        problem_file = f.read()

    return problem_file


ALFRED_LINEARIZED_CODEX_TYPES = """(:otype
        CandleType - otype
        ShowerGlassType - otype
        CDType - otype
        TomatoType - otype
        MirrorType - otype
        ScrubBrushType - otype
        MugType - otype
        ToasterType - otype
        PaintingType - otype
        CellPhoneType - otype
        LadleType - otype
        BreadType - otype
        PotType - otype
        BookType - otype
        TennisRacketType - otype
        ButterKnifeType - otype
        ShowerDoorType - otype
        KeyChainType - otype
        BaseballBatType - otype
        EggType - otype
        PenType - otype
        ForkType - otype
        VaseType - otype
        ClothType - otype
        WindowType - otype
        PencilType - otype
        StatueType - otype
        LightSwitchType - otype
        WatchType - otype
        SpatulaType - otype
        PaperTowelRollType - otype
        FloorLampType - otype
        KettleType - otype
        SoapBottleType - otype
        BootsType - otype
        TowelType - otype
        PillowType - otype
        AlarmClockType - otype
        PotatoType - otype
        ChairType - otype
        PlungerType - otype
        SprayBottleType - otype
        HandTowelType - otype
        BathtubType - otype
        RemoteControlType - otype
        PepperShakerType - otype
        PlateType - otype
        BasketBallType - otype
        DeskLampType - otype
        FootstoolType - otype
        GlassbottleType - otype
        PaperTowelType - otype
        CreditCardType - otype
        PanType - otype
        ToiletPaperType - otype
        SaltShakerType - otype
        PosterType - otype
        ToiletPaperRollType - otype
        LettuceType - otype
        WineBottleType - otype
        KnifeType - otype
        LaundryHamperLidType - otype
        SpoonType - otype
        TissueBoxType - otype
        BowlType - otype
        BoxType - otype
        SoapBarType - otype
        HousePlantType - otype
        NewspaperType - otype
        CupType - otype
        DishSpongeType - otype
        LaptopType - otype
        TelevisionType - otype
        StoveKnobType - otype
        CurtainsType - otype
        BlindsType - otype
        TeddyBearType - otype
        AppleType - otype
        WateringCanType - otype
        SinkType - otype
(:rtype
        ArmChairType - rtype
        BedType - rtype
        BathtubBasinType - rtype
        DresserType - rtype
        SafeType - rtype
        DiningTableType - rtype
        SofaType - rtype
        HandTowelHolderType - rtype
        StoveBurnerType - rtype
        CartType - rtype
        DeskType - rtype
        CoffeeMachineType - rtype
        MicrowaveType - rtype
        ToiletType - rtype
        CounterTopType - rtype
        GarbageCanType - rtype
        CoffeeTableType - rtype
        CabinetType - rtype
        SinkBasinType - rtype
        OttomanType - rtype
        ToiletPaperHangerType - rtype
        TowelHolderType - rtype
        FridgeType - rtype
        DrawerType - rtype
        SideTableType - rtype
        ShelfType - rtype
        LaundryHamperType - rtype
)"""
