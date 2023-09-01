import os
import random
import json
import llm_operators.experiment_utils as experiment_utils
from llm_operators.codex.codex_core import get_solved_unsolved_problems, get_completions
from llm_operators.codex.codex_core import NONE, STOP_TOKEN, CODEX_PROMPT, CODEX_OUTPUT

NATURAL_LANGUAGE_GOAL_START = ";; Goal: "
COT_GOAL_START = ";; Simplified Goal: "
PDDL_GOAL_START = ";; PDDL Goal: "
PDDL_GOAL_REDMINER = ";; Reminder: use ONLY predicates and object types listed in the above PDDL domain. If an English goal contains an object not in the domain, use the most similar available object. All problems are solvable. Propose just ONE goal.\n\n"

DEFAULT_GOAL_TEMPERATURE = 1.0


def propose_goals_for_problems(
    problems,
    domain,
    initial_pddl_predicates,
    supervision_pddl,
    include_codex_types=False,
    temperature=DEFAULT_GOAL_TEMPERATURE,
    n_samples=4,
    max_samples=20,
    use_mock=False,
    use_gt=False,
    print_every=1,
    command_args=None,
    experiment_name='',
    curr_iteration=None,
    output_directory=None,
    resume=False,
    resume_from_iteration=None,
    resume_from_problem_idx=None,
    verbose=False,
):
    random.seed(command_args.random_seed)

    def get_prompt(max_goal_examples=max_samples):
        # Generate unique prompt for each sample
        prompt = nl_header
        if supervision_pddl:  # Add supervision from external prompts.
            prompt += _get_supervision_goal_prompt(supervision_pddl)

        max_goal_examples = min(max_goal_examples, len(solved_problems))
        solved_to_prompt = random.sample(solved_problems, max_goal_examples)

        # domains for all alfred problems should be the same.
        prompt += _get_domain_string(domain, solved_to_prompt[0])
        for solved_problem in solved_to_prompt:  # constructing the input prompt
            prompt += _get_solved_goal_prompt(domain, solved_problem)
        prompt += _get_unsolved_goal_prompt(domain, problem, include_codex_types=include_codex_types, include_domain_string=False)
        return prompt

    unsolved_problems, solved_problems = get_solved_unsolved_problems(problems, context='pddl_goal')
    if use_gt:
        print("Using ground truth goals, skipping: propose_goals_for_problems")
        return
    output_json = {}
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"
    output_filepath = f"{experiment_tag}codex_goals_{'_'.join(initial_pddl_predicates)}.json"
    if resume and os.path.exists(os.path.join(output_directory, output_filepath)):
        mock_propose_goals_for_problems(output_filepath, unsolved_problems, output_directory, domain)
        return
    if use_mock and experiment_utils.should_use_checkpoint(
        curr_iteration=curr_iteration,
        curr_problem_idx=None,
        resume_from_iteration=resume_from_iteration,
        resume_from_problem_idx=resume_from_problem_idx,
    ):
        mock_propose_goals_for_problems(output_filepath, unsolved_problems, output_directory, domain)
        return

    if verbose:
        print(f"propose_goals_for_problems:: proposing for {len(unsolved_problems)} unsolved problems.")

    nl_header = "\n;; Natural language goals and PDDL goals\n\n"

    for idx, problem in enumerate(unsolved_problems):
        # For now, we completely reset the goals if we're proposing.
        problem.proposed_pddl_goals = []
        if verbose and idx % print_every == 0:
            print(f"propose_goals_for_problems:: now on {idx} / {len(unsolved_problems)}")
        try:
            goal_strings = []
            for i in range(n_samples):
                prompt = get_prompt()
                goal_strings.append(get_completions( prompt, temperature=temperature, stop=STOP_TOKEN, n_samples=1)[0])
            output_json[problem.problem_id] = {
                CODEX_PROMPT: prompt,
                CODEX_OUTPUT: goal_strings,
            }
            if verbose:
                print(f'propose_goals_for_problems:: "{problem.language}":')
                for i, goal_string in enumerate(goal_strings):
                    print(f"[Goal {i+1}/{len(goal_strings)}]")
                    print(goal_string)
            problem.proposed_pddl_goals.extend(goal_strings)  # editing the problem
        except Exception as e:
            print(e)
            continue
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)


def mock_propose_goals_for_problems(output_filepath, unsolved_problems, output_directory, current_domain):
    with open(os.path.join(output_directory, output_filepath), "r") as f:
        output_json = json.load(f)
    print(f"mock_propose_goals_for_problems:: from {os.path.join(output_directory, output_filepath)}")
    for p in unsolved_problems:
        if p.problem_id in output_json:
            p.proposed_pddl_goals.extend(output_json[p.problem_id][CODEX_OUTPUT])
    print(
        f"mock_propose_goals_for_problems:: loaded a total of {len([p for p in unsolved_problems if len(p.proposed_pddl_goals) > 0])} goals for {len(unsolved_problems)} unsolved problems."
    )
    return


############################################################################################################

# Utility functions for composing the prompt for goal proposal.


def _get_domain_string(domain, problem):
    """
    problem:
        PDDL.Problem object
    returns:
        prompt
    """
    domain_string = (
        "<START DOMAIN>\n"
        + domain.domain_for_goal_prompting(problem.ground_truth_pddl_problem.ground_truth_pddl_problem_string)
        + "\n<END DOMAIN>\n\n"
    )
    return "\n".join([PDDL_GOAL_REDMINER, domain_string])



def _get_solved_goal_prompt(domain, problem):
    """
    problem:
        PDDL.Problem object
    returns:
        prompt
    """
    NL_goal = NATURAL_LANGUAGE_GOAL_START + "\n" + problem.language + "\n"
    COT = COT_GOAL_START + "\n" + problem.chain_of_thought + "\n" if problem.chain_of_thought else ""
    pddl_goal = PDDL_GOAL_START + "\n" + problem.ground_truth_pddl_problem.ground_truth_goal + "\n" + STOP_TOKEN
    return "\n\n".join([NL_goal, COT, pddl_goal])


def _get_supervision_goal_prompt(supervision_pddl):
    prompt = ""
    for domain_file in supervision_pddl:
        domain = supervision_pddl[domain_file]["domain"]
        pddl_problem_string = supervision_pddl[domain_file]["pddl_problem_string"]
        domain_string = domain.domain_for_goal_prompting(pddl_problem_string)
        NL_goal = NATURAL_LANGUAGE_GOAL_START + "\n" + supervision_pddl[domain_file]["NL_goal"]
        pddl_goal = PDDL_GOAL_START + "\n" + supervision_pddl[domain_file]["goal_pddl"] + "\n" + STOP_TOKEN
        prompt += "\n\n".join([domain_string, NL_goal, pddl_goal])
    return prompt


def _get_unsolved_goal_prompt(domain, problem, include_codex_types=False, include_domain_string=True):
    if include_domain_string:
        domain_string = domain.domain_for_goal_prompting(
            problem.ground_truth_pddl_problem.ground_truth_pddl_problem_string,
            include_codex_types=include_codex_types,
        )
    else:
        domain_string = ""
    NL_goal = "\n" + NATURAL_LANGUAGE_GOAL_START + "\n" + problem.language + "\n"
    return "\n\n".join([domain_string, NL_goal])

