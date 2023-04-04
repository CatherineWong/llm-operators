#### Experiments log.
# 3-29. Tighter task planning / motion planning inner loop with motion planner.
# With ground truth operators.
python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean_38354 --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple pick_clean_then_place_in_recep --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject --verbose --train_iterations 1 --dataset_pddl_directory data/dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_ground_truth_operators --debug_mock_propose_goals
# Running on problem 4 - /Users/catwong/Desktop/zyzzyva/code/llm-operators/data/dataset/alfred_linearized_pddl/train/pick_heat_then_place_in_recep-Egg-None-Fridge-17/trial_T20190909_130835_721326/problem_0.pddl - heating an egg.

# Steps to install.
1. conda create --name llm-operators-38 python=3.8
3. pip install openai pddlgym
4. pip install -r alfred/requirements_python3_8.txt # This requires a specific version of numpy that should be set to numpy==1.21.6 so that it works with scikitimage.
5. pip install lark ipdb tabulate
6. conda install -c conda-forge pytorch==1.13.1
7. Change the site packages installation of ai2thor/server.py so that the `return np.asscalar(obj)` is now `return obj.item()` -- this allows it to run to numpy > 1.16, as required by torch.
8. pip install einops


### Supervision PDDL, alfred_linearized_100, with CleanObject
alfred_linearized_100_supervision_pddl_pick_place_clean_38354:

python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean_38354 --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple pick_clean_then_place_in_recep --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject --verbose --train_iterations 1 --dataset_pddl_directory data/dataset/alfred_linearized_pddl --output_directory generated

Debugging goals:
- Why are we preprocessing away the constants when we preprocess the operators?
Test 1: Let's run with the mock operators.
python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean_38354 --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple pick_clean_then_place_in_recep --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject --verbose --train_iterations 1 --dataset_pddl_directory data/dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_mock_propose_operators --debug_mock_propose_goals

- Why are we not proposing any task plans for the goals?
Test 1: let's try the proposed goals but with the ground truth operators.
python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean_38354 --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple pick_clean_then_place_in_recep --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject --verbose --train_iterations 1 --dataset_pddl_directory data/dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_goals --debug_ground_truth_operators
Test 2: let's try the ground truth goals but the proposed operators.
python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean_38354 --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple pick_clean_then_place_in_recep --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject --verbose --train_iterations 1 --dataset_pddl_directory data/dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_mock_propose_operators --debug_ground_truth_goals
Test 3: let's try reloading the proposed task plans for the ground truth goals and moving into the motion planning part.
python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean_38354 --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple pick_clean_then_place_in_recep --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject --verbose --train_iterations 1 --dataset_pddl_directory data/dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_mock_propose_operators --debug_ground_truth_goals --debug_mock_task_plans

### Supervision PDDL, alfred_linearized_100, with CleanObject
==>  Re-running this will write over the `experiment_name` directory. If you run this command locally, consider adding a timestamp (eg. --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean_3_3`.
```python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple pick_clean_then_place_in_recep --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject --verbose --train_iterations 1 --dataset_pddl_directory data/dataset/alfred_linearized_pddl --output_directory generated```

===> To run this while re-loading the GOALS, run:
`python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple pick_clean_then_place_in_recep --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject --verbose --train_iterations 1 --dataset_pddl_directory data/dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_mock_propose_operators --debug_mock_propose_goals`

==> To run this while re-loading the results of the Codex proposed operators with GROUND TRUTH GOALS, run:
`python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple pick_clean_then_place_in_recep --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject --verbose --train_iterations 1 --dataset_pddl_directory data/dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_mock_propose_operators --debug_ground_truth_goals`

==> To run this while mocking out the task plans and the motion plans and assume they all suceeded.
`python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place_clean --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple pick_clean_then_place_in_recep --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle CleanObject --verbose --train_iterations 2 --dataset_pddl_directory data/dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_mock_propose_operators --debug_mock_propose_goals --debug_skip_task_plans --debug_skip_motion_plans`


### Supervision PDDL, alfred_linearized_100 pick and place.
This top-level command (without any of the checkpointing flags) will run the model on the ```alfred_linearized_100`` dataset of goals, with the GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle operators (pick and place).

```python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_linearized_pddl --output_directory generated```

================================================================
### Supervision PDDL, alfred_linearized_100 pick and place. How many of the Codex-proposed goals could we solve if we used the ground truth operator set? 
```--experiment_name  alfred_linearized_100_codex_goals_gt_operators```

To run:
```python main.py --experiment_name alfred_linearized_100_codex_goals_gt_operators --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_linearized_pddl --output_directory generated --debug_ground_truth_operators --debug_skip_motion_plans```
To debug once its run with the cached goals:
````python main.py --experiment_name alfred_linearized_100_codex_goals_gt_operators --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_linearized_pddl --output_directory generated --debug_ground_truth_operators --debug_skip_motion_plans --debug_mock_propose_goals```




================================================================
#### First end-end run. Supervision PDDL, alfred_linearized_100 pick and place. (Experiment tag: alfred_linearized_100_supervision_pddl_pick_place_1_13_2023)
# alfred_linearized_100_supervision_pddl_pick_place_1_13_2023 - contains the first end-end run from commit: https://github.com/CatherineWong/llm-operators/commit/f2d82771edaad76e3d4559b797fe9d692250ff88 
# Resuming this means you need to resume with 
```--experiment_name  alfred_linearized_100_supervision_pddl_pick_place_1_13_2023 experiment_name```
Original command: ```python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_linearized_pddl --output_directory generated```

Replicate with cached operators and GT goals using:
 ```python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_linearized_pddl --output_directory generated/test_outputs --debug_mock_propose_plans --debug_mock_propose_operators --debug_ground_truth_goals```

Replicate with plan proposal and operators, but with goal proposal:
```python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_mock_propose_operators```

Replicate with cached everything:
```python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place_1_13_2023 --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_mock_propose_operators --debug_mock_propose_goals```


_Codex proposal notes:_
Supervision on external PDDL plans. Initialized only with pick and place.
We now construct a prompt containing handwritten natural language and PDDL plans/operators from external IPC problems. See:
    - `dataset/supervision-NL.json`
    - `supervision-NLgoals-operators.json`. 

- Plan proposal notes: `generated/alfred_linearized_100_supervision_pddl_pick_place_codex_plans.json`: much more diverse but redundant plan set.

Supervision using the 'linearized' pick and place operators in `alfred_linearized.pddl`. Also don't use pick/place, only use GoToLocation. No external PDDL.
- Operator proposal notes: `alfred_linearized_100_supervision_pddl_pick_place_codex_operators.json`

_Task planning notes:_
Planning with the ground truth *linearized* goals.
Replicate with mock data, GT goals, and task plans using:
(NB: only the task plans files are mocked out. There isn't currently a good way to load this.)
 ```python main.py --experiment_name alfred_linearized_100_supervision_ground_truth_goals_pddl_pick_place --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_mock_propose_operators --debug_ground_truth_goals --debug_mock_task_plans```

Planning with the Codex-proposed linearized goals.
 ```python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_mock_propose_operators --debug_ground_truth_goals --debug_mock_task_plans```

_Motion planning notes:_
To skip motion planning (and assume that all task plans succeeded):
 ```python main.py --experiment_name alfred_linearized_100_supervision_pddl_pick_place --dataset_name alfred_linearized_100 --supervision_name supervision --pddl_domain_name alfred_linearized --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix pick_and_place_simple --initial_pddl_operators GotoLocation PickupObjectInReceptacle PickupObjectNotInReceptacle PutObjectInReceptacle PutReceptacleObjectInReceptacle --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_linearized_pddl --output_directory generated --debug_mock_propose_plans --debug_mock_propose_operators --debug_mock_propose_goals --debug_mock_task_plans --debug_skip_motion_plans```

 Otherwise, you will need to run the commmand without `--debug_skip_motion_plans` to run the motion planner.

To run with motion plans, you can run: 