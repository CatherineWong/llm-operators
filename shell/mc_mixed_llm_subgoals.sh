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

# if [[ -z $EXPR_NAME ]]; then
#     echo "EXPR_NAME is not specified."
#     exit 1
# fi

# Run
set -x

python main.py --experiment_name $EXPR_NAME \
  --dataset_name crafting_world_20230913_mixed --dataset_pddl_directory data/dataset/crafting_world_v20230913_mixed \
  --pddl_domain_name crafting_world_teleport --initial_pddl_operators move-to pick-up place-down mine-iron-ore \
  --supervision_name supervision \
  --initial_goal_supervision_prefix SKIP \
  --train_iterations 1 --output_directory generated \
  --goal_propose_include_codex_types --operator_propose_minimum_usage 1 --planner task_planner_pdsketch_onthefly --maximum_operator_arity 7 --n_attempts_to_plan 1 \
  --verbose $@ --llm_propose_task_predicates --external_task_predicates_supervision data/dataset/crafting-world-crafting-task-predicate-supervision_  --resume


