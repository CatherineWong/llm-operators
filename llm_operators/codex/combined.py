from llm_operators.codex.codex_core import get_solved_unsolved_problems
from llm_operators.codex.operator import DEFAULT_OPERATOR_TEMPERATURE, propose_operators_for_problems, use_ground_truth_operators
from llm_operators.codex.plan import DEFAULT_PLAN_TEMPERATURE, propose_plans_for_problems


def propose_plans_operators_for_problems(
    problems,
    domain,
    supervision_pddl,
    n_plan_samples=5,
    n_operator_samples=5,
    plan_temperature=DEFAULT_PLAN_TEMPERATURE,
    operator_temperature=DEFAULT_OPERATOR_TEMPERATURE,
    operator_minimum_usage=2,  # Minimum time the operator was used.
    operator_use_cot=True,
    use_gt=False,
    external_plan_supervision=None,
    external_operator_supervision=None,
    external_operator_sample_with_prompt=True,
    external_operator_names=None,
    command_args=None,
    curr_iteration=None,
    output_directory=None,
    resume=False,
    resume_from_iteration=None,
    resume_from_problem_idx=None,
    debug_skip_propose_operators_after=None,
    debug_skip_propose_plans_after=None,
    verbose=False,
    llm_model=None
):
    unsolved_problems, solved_problems = get_solved_unsolved_problems(problems, context='pddl_plan')
    if use_gt:
        use_ground_truth_operators(domain, verbose)
        return

    # Condition on: NL goals. Propose: PDDL plans.
    propose_plans_for_problems(
        unsolved_problems=unsolved_problems,
        solved_problems=solved_problems,
        current_domain=domain,
        supervision_pddl=supervision_pddl,
        n_samples=n_plan_samples,
        temperature=plan_temperature,
        external_plan_supervision=external_plan_supervision,
        use_mock=command_args.debug_mock_propose_plans,
        experiment_name=command_args.experiment_name,
        curr_iteration=curr_iteration,
        output_directory=output_directory,
        resume=resume,
        resume_from_iteration=resume_from_iteration,
        resume_from_problem_idx=resume_from_problem_idx,
        debug_skip_propose_plans_after=debug_skip_propose_plans_after,
        verbose=verbose,
        llm_model=llm_model,
    )
    # Condition on: new operator names. Propose: PDDL operator definitions.
    propose_operators_for_problems(
        problems=problems,
        current_domain=domain,
        supervision_pddl=supervision_pddl,
        initial_pddl_predicates=command_args.initial_pddl_predicates,
        use_cot=operator_use_cot,
        minimum_usage=operator_minimum_usage,
        temperature=operator_temperature,
        n_samples=n_operator_samples,
        external_operator_supervision=external_operator_supervision,
        external_operator_sample_with_prompt=external_operator_sample_with_prompt,
        external_operator_names=external_operator_names,
        use_mock=command_args.debug_mock_propose_operators,
        experiment_name=command_args.experiment_name,
        curr_iteration=curr_iteration,
        output_directory=output_directory,
        resume=resume,
        resume_from_iteration=resume_from_iteration,
        resume_from_problem_idx=resume_from_problem_idx,
        debug_skip_propose_operators_after=debug_skip_propose_operators_after,
        verbose=verbose,
        llm_model=llm_model,
    )
