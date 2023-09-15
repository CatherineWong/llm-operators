from llm_operators import motion_planner


def _run_llm_propose_task_predicates_motion_planner(pddl_domain, problem_idx, problem_id, planning_problems, args, curr_iteration, output_directory, plan_pass_identifier, plan_attempt_idx, goal_idx, rng):
    problems = planning_problems["train"]

    # Sham task plans following the task planner format.
    task_plans = {
       problems[problem_id].ground_truth_pddl_problem.ground_truth_goal:  problems[problem_id].proposed_pddl_task_predicates
    }
    any_motion_planner_success, new_motion_plan_keys, used_motion_mock = motion_planner.attempt_motion_plan_for_problem(
            pddl_domain=pddl_domain,
            problem_idx=problem_idx,
            problem_id=problem_id,
            problems=planning_problems["train"],
            dataset_name=args.dataset_name,
            new_task_plans=task_plans,
            use_mock=args.debug_mock_motion_plans,
            command_args=args,
            curr_iteration=curr_iteration,
            output_directory=output_directory,
            plan_pass_identifier=plan_pass_identifier,
            plan_attempt_idx=plan_attempt_idx,
            resume=args.resume,
            resume_from_iteration=args.resume_from_iteration,
            resume_from_problem_idx=args.resume_from_problem_idx,
            debug_skip=args.debug_skip_motion_plans,
            verbose=args.verbose,
            llm_propose_task_predicates=args.llm_propose_task_predicates # Baseline -- we skip task proposal if so.
        )

    return any_motion_planner_success
