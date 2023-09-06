"""
prepare_alfred_filtered.py
Allows a human annotator to reject examples of goals based on the oracle goal check.

Rejection criteria:
1. Final location is missing from goal. Example: "get a knife to cut a tomato." - train/pick_and_place_simple-TomatoSliced-None-SinkBasin-5/trial_T20190907_233445_502551
2. Object is missing from goal: Example: "get the knife from the counter then open and close the fridge door." - train/pick_and_place_simple-TomatoSliced-None-Fridge-6/trial_T20190907_233305_019895
3. Key verb is entirely missing from goal, and cannot be inferred from the receptacle. "move cooked food to the kitchen table." - train/pick_heat_then_place_in_recep-BreadSliced-None-DiningTable-7/trial_T20190906_200122_168916 -- this is also an example of entirely missing the core specified object.
    "putting a tomato slice in the fridge." - train/pick_heat_then_place_in_recep-TomatoSliced-None-Fridge-14/trial_T20190911_114912_806806

3. Overspecification of goal (this might make it impossible to execute): Example: "to put a cooled plus sliced tomato on the table to the right of the fridge." - train/pick_and_place_simple-TomatoSliced-None-SideTable-3/trial_T20190908_091658_623792
4. 

Acceptable:
1. Any reasonable synonym for a location: Example: "place a chilled pan in the cupboard." - train/pick_cool_then_place_in_recep-Pan-None-Cabinet-18/trial_T20190907_222321_075263
2. Any reasonable synonym for a verb: Example: "put a microwaved egg in the garbage." -- train/pick_heat_then_place_in_recep-Egg-None-GarbageCan-8/trial_T20190909_120617_866099

"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

import random

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--raw_alfred_goal_file",
    type=str,
    default="data/dataset/alfred-cot-250-filtered-NLgoals-operators_raw.json",
    help="Original ALFRED file.",
)

parser.add_argument(
    "--output_alfred_goal_file",
    type=str,
    default="data/dataset/alfred-cot-250-filtered-NLgoals-operators.json",
    help="Original ALFRED file.",
)
parser.add_argument(
    "--filter", action="store_true", help="Run filtering.",
)

rng = np.random.default_rng(0)

def load_alfred_raw_goals_file(args):
    with open(args.raw_alfred_goal_file) as f:
        raw_goals = json.load(f)
    goal_type_to_goals = defaultdict(lambda: defaultdict(list))
    for split in raw_goals:
        for g in raw_goals[split]:
            goal_type_to_goals[split][g['goal_prefix']].append(g)
    
    for split in goal_type_to_goals:
        print(f"split: {split}")
        for goal_type in goal_type_to_goals[split]:
            print(f"{goal_type} : {len(goal_type_to_goals[split][goal_type])}")
    return goal_type_to_goals
def manually_filter_goals(args, goal_type_to_goals):
    filtered_goal_type_to_goals = defaultdict(lambda: defaultdict(list))
    rejected_goal_type_to_goals = defaultdict(lambda: defaultdict(list))
    for split in goal_type_to_goals:
        for goal_type in goal_type_to_goals[split]:
            for idx, goal in enumerate(goal_type_to_goals[split][goal_type]):
                print(f"{idx}: {goal['cot']}")

                i = ""
                while i not in ["y", "n"]:
                    i = input(f"{idx}: {goal['goal']}")
                category = filtered_goal_type_to_goals if i == "y" else rejected_goal_type_to_goals
                category[split][goal_type].append(goal)
                with open(args.output_alfred_goal_file + "_temp.json", "w") as f:
                    json.dump({
                        "accepted" : filtered_goal_type_to_goals,
                        "rejected" : rejected_goal_type_to_goals
                    }, f)

def downsample_goals(args, original_goal_type_to_goals):
    MAX_GOALS = 250
    with open(args.output_alfred_goal_file + "_temp.json") as f:
        filtered_goals = json.load(f)
        accepted = filtered_goals['accepted']
    
    downsampled_goal_type_to_goals = defaultdict(lambda: defaultdict())
    for split in original_goal_type_to_goals:
        total_goals = np.sum([len(original_goal_type_to_goals[split][g]) for g in original_goal_type_to_goals[split]])
        for goal_type in original_goal_type_to_goals[split]:
            total_to_sample = int(float(len(original_goal_type_to_goals[split][goal_type]) / total_goals) * MAX_GOALS)
            print(f"Sampling {split} {goal_type}: {total_to_sample} of possible {len(accepted[split][goal_type])}")

            sampled = rng.choice(accepted[split][goal_type], min(len(accepted[split][goal_type]), total_to_sample), replace=False)
            downsampled_goal_type_to_goals[split][goal_type] = list(sampled)      
        print(f"Total downsampled: {np.sum([len(downsampled_goal_type_to_goals[split][g]) for g in downsampled_goal_type_to_goals[split]])}")  
    # Flatten
    flattened_down_sampled_goals = defaultdict(list)
    for split in downsampled_goal_type_to_goals:
        for goal_type in downsampled_goal_type_to_goals[split]:
            flattened_down_sampled_goals[split] += downsampled_goal_type_to_goals[split][goal_type]

    with open(args.output_alfred_goal_file, "w") as f:
        json.dump(flattened_down_sampled_goals, f)

def main():
    args = parser.parse_args()
    alfred_nl_goal_type_to_goals = load_alfred_raw_goals_file(args)
    if args.filter:
        manually_filter_goals(args, alfred_nl_goal_type_to_goals)
    downsample_goals(args, alfred_nl_goal_type_to_goals)




if __name__ == "__main__":
    main()
