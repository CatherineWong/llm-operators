#! /bin/bash
#
# mc_02024.sh
# Copyright (C) 2023 Jiayuan Mao <maojiayuan@gmail.com>
#
# Distributed under terms of the MIT license.
#

set -e

# Usage:: <mc_0204.sh> --expr EXPR_NAME

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --expr)
            EXPR_NAME="$2"
            shift
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

if [[ -z $EXPR_NAME ]]; then
    echo "EXPR_NAME is not specified."
    exit 1
fi

# Run
set -x

python main.py --experiment_name $EXPR_NAME \
  --dataset_name crafting_world_20230829_crafting_only --dataset_pddl_directory data/dataset/crafting_world_v20230829_crafting_only \
  --pddl_domain_name crafting_world_teleport --initial_pddl_operators move-to pick-up place-down craft-wood-plank craft-arrow \
  --supervision_name supervision \
  --initial_goal_supervision_prefix SKIP \
  --train_iterations 5 --output_directory generated --operator-use-cot 1 \
  --goal_propose_include_codex_types --operator_propose_minimum_usage 1 --planner task_planner_pdsketch_onthefly --maximum_operator_arity 9 \
  --n_attempts_to_plan 3 --n_goal_samples 1 --n_plan_sample 1 \
  --external_operator_supervision data/dataset/crafting-world-crafting-only-operator-supervision_ --external_operator_sample_with_prompt --external_operator_names craft-arrow craft-wood-plank \
  --planner_timeout 120 \
  --verbose $@
