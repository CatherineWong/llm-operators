

def evaluate_task_plans_and_costs_for_problems(
    pddl_domain,
    problems,
    command_args,
    verbose=False,
    output_directory=None,
    use_mock=False,
    debug_skip=False,
    proposed_operators: Optional[Sequence[str]] = None,
    task_plan_with_constants=False,
    max_task_samples=4,
):
    """
    Batch evaluates planner to evaluate task plans for a set of planning problems, given a PDDL domain.

    For now, this just runs using the first operator definition.
    :ret: problems updated with PDDL plans.
    """
    if debug_skip:
        print(f"debug_skip_task_plans_and_costs_for_problems on {len(problems)}.")
        return
    print(f"evaluate_task_plans_and_costs_for_problems on {len(problems)}.")

    output_json = []
    experiment_tag = "" if len(command_args.experiment_name) < 1 else f"{command_args.experiment_name}_"

    output_filepath = f"{experiment_tag}task_plans.json"

    if use_mock:
        mock_evaluate_task_plans_and_costs_for_problems(output_filepath, output_directory, problems)
        return

    if verbose:
        print(f"Use ground truth goals? {command_args.debug_ground_truth_goals}")

    total_solved_problems = 0
    for max_problems, problem_id in enumerate(problems):
        if verbose:
            print(
                f"\nNow on problem {max_problems} / {len(problems)}. Total solved problems so far: {total_solved_problems}"
            )
        any_success, new_evaluated_plans, problem_json = sample_task_plans_for_problem(
            pddl_domain=pddl_domain,
            problem=problems[problem_id],
            planner_type=command_args.planner,
            command_args=command_args,
            verbose=verbose,
            output_directory=output_directory,
            debug_ground_truth_goals=command_args.debug_ground_truth_goals,
            proposed_operators=proposed_operators,
            max_task_samples=max_task_samples,
        )
        problems[problem_id].update_evaluated_pddl_plans(new_evaluated_plans)
        if any_success:
            total_solved_problems += 1
        output_json.append(problem_json)
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)


def get_top_k_operators_to_cover_operator_downsampling_percentage(
    categorical_operator_scores, operator_downsampling_percentage
):
    # If we greedily chose the operators,
    sorted_scores = sorted(categorical_operator_scores, reverse=True)
    total_percentage = 0
    for idx, score in enumerate(sorted_scores):
        total_percentage += score
        if total_percentage >= operator_downsampling_percentage:
            return idx + 1