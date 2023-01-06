# Mock with all initial predicates.
python main.py --dataset_name alfred --pddl_domain_name alfworld --dataset_fraction 0.001 --training_plans_fraction 0.1 --initial_pddl_operators GotoLocation PickupObject PutObject  --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_pddl --output_directory generated/test_outputs --debug_mock_propose_plans --debug_mock_propose_operators
# Mock with predicate learning.
python main.py --dataset_name alfred --pddl_domain_name alfworld --dataset_fraction 0.001 --training_plans_fraction 0.1 --initial_pddl_operators GotoLocation PickupObject PutObject --initial_pddl_predicates NONE  --verbose --train_iterations 1 --dataset_pddl_directory dataset/alfred_pddl --output_directory generated/test_outputs --debug_mock_propose_plans 