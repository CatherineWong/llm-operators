import os
import random
import json
import llm_operators.experiment_utils as experiment_utils
from llm_operators.codex.codex_core import get_solved_unsolved_problems, get_completions
from llm_operators.codex.codex_core import NONE, STOP_TOKEN, CODEX_PROMPT, CODEX_OUTPUT

# v0
NATURAL_LANGUAGE_GOAL_START = ";; Goal: "
COT_GOAL_START = ";; Simplified Goal: "
PDDL_GOAL_START = ";; PDDL Goal: "
PDDL_GOAL_REDMINER = ";; Reminder: use ONLY predicates and object types listed in the above PDDL domain. If an English goal contains an object not in the domain, use the most similar available object. All problems are solvable. Propose just ONE goal.\n\n"

# v1
NATURAL_LANGUAGE_GOAL_START_V1 = ";; Human written natural language goal."
PDDL_GOAL_START_V1 = ";; PDDL guess 1 for '{language}'"
COT_GOAL_START_V1 = ";; Official goal name: "

DEFAULT_GOAL_TEMPERATURE = 1.0

GOAL_SAMPLING_START_TOKEN = "<START>"
GOAL_SAMPLING_END_TOKEN = "<END>"


def propose_goals_for_problems(
    dataset_name,
    problems,
    domain,
    initial_pddl_predicates,
    supervision_pddl,
    include_codex_types=False,
    temperature=DEFAULT_GOAL_TEMPERATURE,
    n_samples=4,
    max_goal_examples=10,
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
    external_goal_supervision=None,
    external_goal_sample_with_prompt=False,
):
    random.seed(command_args.random_seed)

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

    for idx, problem in enumerate(unsolved_problems):
        # For now, we completely reset the goals if we're proposing.
        problem.proposed_pddl_goals = []
        if verbose and idx % print_every == 0:
            print(f"propose_goals_for_problems:: now on {idx} / {len(unsolved_problems)}")

        #### LCW - Temporary split between ALFRED and Minecraft behaviors to preserve exact prompting behaviors used in ICML experiments.
        if "alfred" in dataset_name:
            ### ALFRED goal prompting.
            codex_prompt, proposed_goal_definitions = _propose_alfred_goal_definition(domain, solved_problems, problem, n_samples, temperature, include_codex_types, max_goal_examples, external_goal_supervision, external_goal_sample_with_prompt)

            output_json[problem.problem_id] = {
                CODEX_PROMPT: codex_prompt,
                CODEX_OUTPUT: proposed_goal_definitions,
            }
            if verbose:
                print(f'propose_goals_for_problems:: "{problem.language}":')
                for i, goal_string in enumerate(proposed_goal_definitions):
                    print(f"[Goal {i+1}/{len(proposed_goal_definitions)}]")
                    print(goal_string)
            problem.proposed_pddl_goals.extend(proposed_goal_definitions)  # editing the problem
        else:
            #### Minecraft goal prompting.
            try:
                goal_strings = []
                for i in range(n_samples):
                    prompt = _get_minecraft_prompt(max_goal_examples, supervision_pddl, domain, include_codex_types, problem, solved_problems)
                    goal_strings.append(get_completions(prompt, temperature=temperature, stop=STOP_TOKEN, n_samples=1)[0])
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
def _get_minecraft_prompt(max_goal_examples, supervision_pddl, domain, include_codex_types, problem, solved_problems):
        nl_header = "\n;; Natural language goals and PDDL goals\n\n"
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

# Utility functions for composing the prompt for goal proposal.
def _propose_alfred_goal_definition(domain, solved_problems, problem, n_goal_samples, temperature, include_codex_types, max_goal_examples, external_goal_supervision, external_goal_sample_with_prompt):
    if external_goal_supervision is not None:
         # For now, we also only support sampling with the prompt.
         assert external_goal_sample_with_prompt
         return _propose_goal_definition_external_supervision_sample_with_prompt(
             domain=domain,
             solved_problems=solved_problems,
             problem=problem,
             n_goal_samples=n_goal_samples,
             temperature=temperature,
             max_goal_examples=max_goal_examples,
             external_goal_supervision=external_goal_supervision,
             external_goal_sample_with_prompt=external_goal_sample_with_prompt
         )


    def get_prompt(max_goal_examples=max_goal_examples):
        nl_header = "\n;; Natural language goals and PDDL goals\n\n"
        # Generate unique prompt for each sample
        prompt = nl_header

        max_goal_examples = min(max_goal_examples, len(solved_problems))
        solved_to_prompt = random.sample(solved_problems, max_goal_examples)
        
        # domains for all alfred problems should be the same.
        prompt += _get_domain_string(domain, solved_to_prompt[0])
        for solved_problem in solved_to_prompt:  # constructing the input prompt 
            prompt += _get_solved_goal_prompt(domain, solved_problem)
        prompt += _get_unsolved_goal_prompt(domain, problem, include_codex_types=include_codex_types, include_domain_string=False)
        return prompt
    
    goal_strings = []
    for i in range(n_goal_samples):
        codex_prompt = get_prompt()
        goal_strings.append(get_completions(codex_prompt, temperature=temperature, stop=STOP_TOKEN, n_samples=1)[0])
    return codex_prompt, goal_strings


def _propose_goal_definition_external_supervision_sample_with_prompt(domain, solved_problems, problem, n_goal_samples, temperature, max_goal_examples, external_goal_supervision, external_goal_sample_with_prompt):
    from num2words import num2words

    with open(external_goal_supervision + "system.txt") as f:
        system_message = f.read()
    
    # Add N examples from the current solved problems.
    max_goal_examples = min(max_goal_examples, len(solved_problems))
    solved_to_prompt = random.sample(solved_problems, max_goal_examples)

    for solved_problem in solved_to_prompt:  # constructing the input prompt
         system_message += _get_solved_goal_prompt_v1(domain,solved_problem, natural_language_goal_start="Human written natural language goal", cot_goal_start=COT_GOAL_START, pddl_goal_start=PDDL_GOAL_START)

    with open(external_goal_supervision + "user.txt") as f:
        sampling_message = f.read()
        N_SAMPLES_MASK = "<N_SAMPLES>"
        assert N_SAMPLES_MASK in sampling_message
        sampling_message = sampling_message.replace(N_SAMPLES_MASK, num2words(n_goal_samples))
        sampling_message += problem.language + "\n"
    codex_prompt = [{"role": "system", "content": system_message}, {"role": "user", "content": sampling_message}]
    try:
        completion = get_completions(
            codex_prompt,
            temperature=temperature,
            n_samples=1,
            max_tokens=1500,
        )[0]
        if not external_goal_sample_with_prompt:
            assert False
        # Parse the tokens out of the completion.
        import re

        operator_matches = re.findall(
            rf"{GOAL_SAMPLING_START_TOKEN}(.*?){GOAL_SAMPLING_END_TOKEN}", completion, re.DOTALL
        )[:n_goal_samples]
        return codex_prompt, operator_matches
    except:
        return codex_prompt, []

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



def _get_solved_goal_prompt(domain, problem, natural_language_goal_start=NATURAL_LANGUAGE_GOAL_START, cot_goal_start=COT_GOAL_START, pddl_goal_start=PDDL_GOAL_START, format_pddl_goal_start=False):
    """
    problem:
        PDDL.Problem object
    returns:
        prompt
    """
    NL_goal = natural_language_goal_start + "\n" + problem.language + "\n"
    COT = cot_goal_start + "\n" + problem.chain_of_thought + "\n" if problem.chain_of_thought else ""
    pddl_goal = PDDL_GOAL_START + "\n" + problem.ground_truth_pddl_problem.ground_truth_goal + "\n" + STOP_TOKEN
    return "\n\n".join([NL_goal, COT, pddl_goal])

def _get_solved_goal_prompt_v1(domain, problem, natural_language_goal_start=NATURAL_LANGUAGE_GOAL_START_V1, cot_goal_start=COT_GOAL_START_V1, pddl_goal_start=PDDL_GOAL_START_V1, format_pddl_goal_start=True):
    """
    subtly different solved goal prompt for prompts with multiple guesses. See 'alfred-goal-supervision_system.txt' for examples.

    ;; Human written natural language goal.
    put a warmed up slice of pear in the cabinet.

    ;; PDDL guess 1 for "put a warmed up slice of pear in the cabinet."
    ;; Official goal name: pick_heat_then_place_in_recep-AppleSliced-None-Cabinet
    (:goal
            (exists (?r - receptacle)
            (exists (?o - object)
                (and 
                    (objectType ?o AppleType) 
                    (receptacleType ?r CabinetType)
                    (inReceptacle ?o ?r)
                    (heatable ?o)
                    (isHot ?o)
                    (sliceable ?o)
                    (isSliced ?o)  
                )
        )))

    problem:
        PDDL.Problem object
    returns:
        prompt
    """
    NL_goal = natural_language_goal_start + "\n" + problem.language 
    pddl_guess = PDDL_GOAL_START_V1.format(language=problem.language)
    cot = cot_goal_start + "\n" + problem.chain_of_thought + "\n" if problem.chain_of_thought else ""
    pddl_goal = problem.ground_truth_pddl_problem.ground_truth_goal + "\n" + STOP_TOKEN
    return "\n".join([NL_goal, pddl_guess, cot, pddl_goal])

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

