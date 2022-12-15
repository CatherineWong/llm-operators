"""
motion_planner.py
Utilities for generating motion plans.
"""
from pddl import PDDLPlan
from alfred.alfredplanner import init_alfred, search, Literal, Fluent


def evaluate_motion_plans_and_costs_for_problems(
    pddl_domain,
    problems,
    command_args,
    verbose=False,
    output_directory=None,
    use_mock=False,
    dataset_name="",
):
    """
    Runs a motion planner.
    """
    if "alfred" in dataset_name:
        evaluate_alfred_motion_plans_and_costs_for_problems(
            pddl_domain,
            problems,
            command_args,
            verbose=verbose,
            output_directory=output_directory,
            use_mock=use_mock,
        )
    else:
        print(f"Unsupported dataset name: {dataset_name}")
        assert False


def mock_alfred_motion_plans_and_costs_for_problems(
    output_filepath, output_directory, problems
):
    assert False


def evaluate_alfred_motion_plans_and_costs_for_problems(
    pddl_domain,
    problems,
    command_args,
    verbose=False,
    output_directory=None,
    use_mock=False,
):
    print(f"evaluate_motion_plans_and_costs_for_problems on {len(problems)} problems.")
    output_json = []
    experiment_tag = (
        ""
        if len(command_args.experiment_name) < 1
        else f"{command_args.experiment_name}_"
    )
    output_filepath = f"{experiment_tag}motion_plans.json"

    if use_mock:
        mock_alfred_motion_plans_and_costs_for_problems(
            output_filepath, output_directory, problems
        )
        return
    for max_problems, problem_id in enumerate(problems):
        for pddl_goal in problems[problem_id].evaluated_pddl_plans:
            plan = problems[problem_id].evaluated_pddl_plans[pddl_goal]
            if plan is not None and plan != {}:
                if verbose:
                    print(f"Motion planning for: {problem_id}")
                    print(f"Proposed goal is: ")
                    print(pddl_goal)
                    print(f"Ground truth oracle goal is: ")
                    print(
                        problems[problem_id].ground_truth_pddl_problem.ground_truth_goal
                    )
                    attempt_sequential_plan_alfred(plan, pddl_domain, verbose)


def preprocess_alfred_action_arg(action_arg):
    from alfred.gen.utils.py_util import multireplace

    action_arg = multireplace(
        action_arg,
        {
            "_minus_": "-",
            "-": "#",
            "_bar_": "|",
            "_plus_": "+",
            "_dot_": ".",
            "_comma_": ",",
        },
    )
    return action_arg


def attempt_sequential_plan_alfred(pddl_plan, pddl_domain, verbose=False):
    """"pddl_plan: a PDDLPlan object with operators on it."""
    for action in pddl_plan.plan:
        print("Attempting to ex")
        ground_postcondition_predicates = PDDLPlan.get_postcondition_predicates(
            action, pddl_domain
        )
        ground_postcondition_fluents = [
            Literal(
                fluent=Fluent(
                    predicate=predicate.name,
                    objects=(
                        preprocess_alfred_action_arg(action_arg)
                        for action_arg in predicate.argument_values
                    ),
                ),
                neg=predicate.neg,
            )
            for predicate in ground_postcondition_predicates
        ]
        import pdb

        pdb.set_trace()
