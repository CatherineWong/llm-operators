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

python main-cw-manual-solutions.py \
  --dataset_name crafting_world_20230913_mixed --dataset_pddl_directory data/dataset/crafting_world_v20230913_mixed \
  --pddl_domain_name crafting_world_teleport \
  --initial_goal_supervision_prefix SKIP \
  --verbose $@
