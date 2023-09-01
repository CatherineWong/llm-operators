from llm_operators.codex.codex_core import get_solved_unsolved_problems
from llm_operators.codex.plan import propose_plans_for_problems
from llm_operators.codex.operator import propose_operators_for_problems
from llm_operators.codex.goal import propose_goals_for_problems
NLgoals_PDDLplans_prompt = "\n;; Natural language goals and PDDL plans\n\n"


def propose_plans_operators_goals_for_problems(
    current_domain,
    problems,
    supervision_pddl=[],
    n_plan_samples=5,
    n_operator_samples=5,
    temperature=0.0,
    verbose=False,
    output_directory=None,
    command_args=None,
    external_plan_supervision=None,
):
    """
    Proposes PDDL operators, goals, and plans for unsolved problems using Codex.
    Problems are updated with proposed plans and goals.
    ret:
        proposed_codex_operators: PDDL operators proposed by Codex.
    """
    unsolved_problems, solved_problems = get_solved_unsolved_problems(problems, context='pddl_plan')
    if verbose:
        print("Now in: propose_plans_operators_goals_for_problems: ")
        print(f"\t{len(unsolved_problems)} unsolved problems / {len(solved_problems)} solved problems")

    # Condition on: NL goals. Propose: PDDL plans.
    propose_plans_for_problems(
        unsolved_problems=unsolved_problems,
        solved_problems=solved_problems,
        current_domain=current_domain,
        supervision_pddl=supervision_pddl,
        n_samples=n_plan_samples,
        temperature=temperature,
        verbose=verbose,
        output_directory=output_directory,
        experiment_name=command_args.experiment_name,
        use_mock=command_args.debug_mock_propose_plans,
        external_plan_supervision=external_plan_supervision,
    )

    # Condition on: new operator names. Propose: PDDL operator definitions.
    propose_operators_for_problems(
        problems=problems,
        current_domain=current_domain,
        supervision_pddl=supervision_pddl,
        verbose=verbose,
        temperature=temperature,
        n_samples=n_operator_samples,
        output_directory=output_directory,
        initial_pddl_predicates=command_args.initial_pddl_predicates,
        experiment_name=command_args.experiment_name,
        use_mock=command_args.debug_mock_propose_operators,
    )

    # Condition on: NL goals. Propose: PDDL goals.
    propose_goals_for_problems(
        problems=problems,
        domain=current_domain,
        output_directory=output_directory,
        supervision_pddl=supervision_pddl,
        verbose=verbose,
        temperature=temperature,
        initial_pddl_predicates=command_args.initial_pddl_predicates,
        experiment_name=command_args.experiment_name,
        use_mock=command_args.debug_mock_propose_goals,
        use_gt=command_args.debug_ground_truth_goals,
    )


def get_custom_codex_prompt(solved_problems):
    """
    Hand selects solved problems to use as prompts for Codex.
    Proof-of-concept until we implement a better heuristic for choosing codex prompts.
    Works on alfred-solvable-200 dataset.
    """
    for i, problem in enumerate(solved_problems):
        print(f"[{i}/{len(solved_problems)}]")
        print(problem.language)
        print(problem.ground_truth_pddl_problem.ground_truth_goal)
        print()

    problem_idxs = [0, 4, 7, 9, 12, 18, 22]

    return [p for i, p in enumerate(solved_problems) if i in problem_idxs]


def get_solved_unsolved_problems_or_supervision(problems):
    raise RuntimeError("get_solved_unsolved_problems_or_supervision should not be called. Use get_solved_unsolved_problems.")

    # keep this for now, but should be removed.
    # unsolved_problems = [
    #     problems[p]
    #     for p in problems
    #     if len(problems[p].solved_motion_plan_results) < 1 and not problems[p].should_supervise_pddl_goal
    # ]
    # solved_problems = [
    #     problems[p]
    #     for p in problems
    #     if (len(problems[p].solved_motion_plan_results) > 0) or problems[p].should_supervise_pddl_goal
    # ]
    # return unsolved_problems, solved_problems


def propose_predicates_for_problems(problems, current_domain, use_mock):
    # TBD: to be implemented.
    pass
