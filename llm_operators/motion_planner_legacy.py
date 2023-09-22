
def evaluate_motion_plans_and_costs_for_problems(
    curr_iteration,
    pddl_domain,
    problems,
    command_args,
    verbose=False,
    output_directory=None,
    use_mock=False,
    debug_skip=False,
    dataset_name="",
):
    """
    Runs a motion planner.
    """
    if "alfred" in dataset_name:
        evaluate_alfred_motion_plans_and_costs_for_problems(
            curr_iteration,
            pddl_domain,
            problems,
            command_args,
            verbose=verbose,
            output_directory=output_directory,
            use_mock=use_mock,
            debug_skip=debug_skip,
        )
    elif dataset_name == "crafting_world_20230204_minining_only" or dataset_name == "crafting_world_20230829_crafting_only":
        evaluate_cw_motion_plans_and_costs_for_problems(
            curr_iteration,
            pddl_domain,
            problems,
            command_args,
            verbose=verbose,
            output_directory=output_directory,
            use_mock=use_mock,
            debug_skip=debug_skip,
        )
    else:
        print(f"Unsupported dataset name: {dataset_name}")
        assert False


def evaluate_alfred_motion_plans_and_costs_for_problems(
    curr_iteration,
    pddl_domain,
    problems,
    command_args,
    verbose=False,
    output_directory=None,
    use_mock=False,
    debug_skip=False,
):
    print(f"evaluate_motion_plans_and_costs_for_problems on {len(problems)} problems.")
    experiment_tag = "" if len(command_args.experiment_name) < 1 else f"{command_args.experiment_name}_"
    output_filepath = f"{experiment_tag}motion_plans.json"

    for max_problems, problem_id in enumerate(problems):
        for pddl_goal in problems[problem_id].evaluated_pddl_plans:
            for pddl_plan in problems[problem_id].evaluated_pddl_plans[pddl_goal]:
                # run_motion_planner
                # not implemented envname to task_id
                # recording videos. quicktime player.
                motion_plan_result = evaluate_alfred_motion_plans_and_costs_for_goal_plan(
                    problem_id,
                    problems,
                    pddl_goal,
                    pddl_plan,
                    pddl_domain,
                    verbose,
                    debug_skip=debug_skip,
                )
                problems[problem_id].evaluated_motion_planner_results[pddl_goal] = motion_plan_result
                if motion_plan_result.task_success:
                    problems[problem_id].best_evaluated_plan_at_iteration = curr_iteration
                if verbose:
                    print(f"Motion plan result: task_success: {motion_plan_result.task_success}")
                    if motion_plan_result.last_failed_operator:
                        print(
                            f"Successfully executed: {motion_plan_result.last_failed_operator + 1} / {len(motion_plan_result.pddl_plan)} operators in final task sequence.\n\n\n"
                        )



def evaluate_cw_motion_plans_and_costs_for_problems(
    curr_iteration,
    pddl_domain,
    problems,
    command_args,
    verbose=False,
    output_directory=None,
    use_mock=False,
    debug_skip=False,
):
    print(f"evaluate_motion_plans_and_costs_for_problems on {len(problems)} problems.")
    experiment_tag = "" if len(command_args.experiment_name) < 1 else f"{command_args.experiment_name}_"
    output_filepath = f"{experiment_tag}motion_plans.json"

    if use_mock:
        # Not implemented
        assert False

    for max_problems, problem_id in enumerate(problems):
        for pddl_goal in problems[problem_id].evaluated_pddl_plans:
            pddl_plan = problems[problem_id].evaluated_pddl_plans[pddl_goal]
            if pddl_plan is not None and pddl_plan != {} and pddl_plan.plan is not None:
                motion_plan_result = evaluate_cw_motion_plans_and_costs_for_goal_plan(
                    # current_domain_string, current_problem_string, pddl_goal, pddl_plan
                    problem_id,
                    problems,
                    pddl_goal,
                    pddl_plan,
                    pddl_domain,
                    verbose,
                    debug_skip=debug_skip,
                )
                problems[problem_id].evaluated_motion_planner_results[pddl_goal] = motion_plan_result
                if motion_plan_result.task_success:
                    problems[problem_id].best_evaluated_plan_at_iteration = curr_iteration
                if verbose:
                    print(f"Motion plan result: task_success: {motion_plan_result.task_success}")
                    print(
                        f"Successfully executed: {motion_plan_result.last_failed_operator} / {len(motion_plan_result.pddl_plan.plan)} operators in task plan.\n\n\n"
                    )
