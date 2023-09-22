# Minecraft experiments log.

### 4-4 Experiments using a teleport domain

> python main.py --experiment_name cw_v20230204_mining_only_full_20230226_copytoteleport --dataset_name crafting_world_20230204_minining_only --supervision_name supervision --pddl_domain_name crafting_world_teleport --training_plans_fraction 1.0 --initial_plans_prefix mining --initial_pddl_operators move-to pick-up place-down mine-iron-ore --train_iterations 1 --dataset_pddl_directory data/dataset/crafting_world_v202302024_mining_only --goal_propose_include_codex_types --operator_propose_minimum_usage 1 --output_directory generated --planner task_planner_pdsketch_onthefly --verbose --debug_mock_propose_plans --debug_mock_propose_operators --debug_mock_propose_goals --maximum_operator_arity 7

### 3-29 Experiments with GPT-4

> python main.py --experiment_name cw_v20230204_mining_only_full_20230329 --dataset_name crafting_world_20230204_minining_only --supervision_name supervision --pddl_domain_name crafting_world --training_plans_fraction 1.0 --initial_plans_prefix mining --initial_pddl_operators move-up move-down move-left move-right pick-up place-down mine-iron-ore --train_iterations 2 --dataset_pddl_directory data/dataset/crafting_world_v202302024_mining_only --goal_propose_include_codex_types --operator_propose_minimum_usage 1 --output_directory generated --planner task_planner_pdsketch_onthefly --verbose --maximum_operator_arity 7

### 3-24: Tighter task planning / motion planning inner loop.

> python main.py --experiment_name cw_v20230204_mining_only_full_20230226 --dataset_name crafting_world_20230204_minining_only --supervision_name supervision --pddl_domain_name crafting_world --training_plans_fraction 1.0 --initial_plans_prefix mining --initial_pddl_operators move-up move-down move-left move-right pick-up place-down mine-iron-ore --train_iterations 5 --dataset_pddl_directory data/dataset/crafting_world_v202302024_mining_only --goal_propose_include_codex_types --operator_propose_minimum_usage 1 --output_directory generated --planner task_planner_pdsketch_onthefly --verbose --debug_mock_propose_plans --debug_mock_propose_operators --debug_mock_propose_goals --maximum_operator_arity 7 --debug_mock_task_plans --debug_mock_motion_plans

Notes to install:

- Download Jiayuan Concepts github;
- Download Jiayuan Jacinle from github;
- To install torch - you may need to brew install libomp on a Mac.

### From scratch installation:

1. conda create --name llm-operators-39 python=3.9
2. conda install pytorch
3. pip install openai pddlgym
4. pip install -r alfred/requirements_python3_8.txt # This one may not work.
5. pip install lark ipdb tabulate
