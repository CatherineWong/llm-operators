import code

import planning_domain
import openai
import os
from time import sleep
import random
from pddl_parser import *

random.seed(0)

OPERATOR_START = ";; Operator: "
EXAMPLE_START = ";; Example: "
OPERATOR_START_TOKEN = "(:action "
OPERATOR_STOP_TOKEN = "\n<END>\n"
NL_PROMPT = "\n#### Natural language goals and PDDL plans\n\n"

# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError(
#         "OPENAI_API_KEY is not set. Please set this in the shell via `export OPENAI_API_KEY=...`"
#     )
# openai.api_key = os.environ["OPENAI_API_KEY"]

openai.api_key = "sk-kXXSnnSNUWZOfDHWRow4edlBSKjeQEFZ7wVASMzS"


def get_completions(prompt, temperature, stop, n_samples=1):
    response = openai.Completion.create(
        engine="code-davinci-002",
        prompt=prompt,
        temperature=0.1,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop,
        n=n_samples,
    )
    return [c["text"] for c in response.choices]


def propose_operators_for_problems(
    current_domain, problems, n_samples=1, verbose=False
):
    # TODO (cw / nk): pseudocode demonstrating usage.
    unsolved_problems = [p for p in problems if p.pddl_plan == None]
    solved_problems = [p for p in problems if p.pddl_plan is not None]
    existing_operator_uses = sample_existing_operator_uses(problems)
    proposed_operator_uses = propose_operator_uses(unsolved_problems, current_domain)

    operator_uses = {**existing_operator_uses, **proposed_operator_uses}
    proposed_operators = [
        p for p in proposed_operator_uses if p not in current_domain.operators
    ]
    proposed_operator_definitions = dict()
    for o in proposed_operators:
        propose_operator_definition(
            current_domain,
            o,
            operator_uses=operator_uses,
            max_operator_examples=10,
            temperature=0.0,
            n_samples=n_samples,
            verbose=False,
        )
    return proposed_operator_definitions


def sample_existing_operator_uses():
    # Get example usages from the solved operator plans.
    # TODO (cw/nk)
    pass


def get_solved_problem_text(problem):
    """
    :param: solved Problem object
    return:
        string to add to the codex input prompt
    """
    problem_text = "#" + problem.language + "\n" + problem.pddl_plan + OPERATOR_STOP_TOKEN
    return problem_text


def propose_operator_uses(unsolved_problems, solved_problems, current_domain,n_samples =1):
    """
    unsolved_problems:
        list of Problem objects to be solved
    solved_problems:
        list of Problem objects with solutions
    current_domain:
        Domain object describing the domain

    edits the unsolved problem objects - adds plans to the problem.proposed_pddl_plan list

    return:
        USES, dict with operator names as keys, and list of example uses as keys

    """
    ## TODO (cw/nk): propose operator names from a set of natural language plans and existing operator / domain definitions.

    # USES: {
    #     "WashObject" : ["(WashObject agent1 loc1 chicken)", "(WashObject agent1 loc1 chicken)"] # from all plans, extract all times it was used
    # }
    prompt = current_domain.to_string() + NL_PROMPT
    USES = {}

    for solved_problem in solved_problems: #constructing the input prompt
        prompt += get_solved_problem_text(solved_problem)

    for problem in unsolved_problems:
        prompt += "\n# " + problem.language
        plan = get_completions(prompt,temperature=0.1,stop=OPERATOR_STOP_TOKEN)[0]
        print(problem.language, " ",plan)
        prompt += plan + OPERATOR_STOP_TOKEN



def  propose_operator_uses_for_problem(unsolved_problem, solved_problems, current_domain):
    # :ret:
    pass

def propose_operator_definition(
    current_domain,
    operator_name_to_define,
    operator_uses={},
    max_operator_examples=10,
    temperature=0.0,
    n_samples=1,
    verbose=False,
):
    """
    Proposes an operator definition for a given domain, and optionally with examples of operator usages.
    current_domain: an existing PDDL domain.
    operator_uses: dict {operator_name: list of string uses of a given operator in PDDL plans.}
    operator_name_to_define: string name of operator to define.

    :ret: list of up to n_samples operator definitions. Empty list if prompting fails.
    """
    if verbose:
        print(
            f"propose_operator_definition: operator_name_to_define - {operator_name_to_define}"
        )
    # Codex prompt header.
    nl_header = (
        ";;;; Define planning operators based on a PDDL domain and example usages.\n\n"
    )
    pddl_domain = (
        ";;;; PDDL domain definition.\n"
        + current_domain.domain_definition_to_string()
        + "\n\n"
    )
    translation_header = ";;;; Define operators based on examples of their usage and the PDDL domain definition above. Only use predicates and functions available in the PDDL domain.\n\n"

    codex_prompt = nl_header + pddl_domain + translation_header

    # Codex prompt exampler operators.
    operator_examples = random.sample(
        list(current_domain.operators.keys()),
        min(len(current_domain.operators), max_operator_examples),
    )
    for o in operator_examples:
        codex_prompt += f"{OPERATOR_START}{o}\n"
        if o in operator_uses:
            for use_example in operator_uses[o]:
                codex_prompt += f"{EXAMPLE_START}{use_example}\n"
        codex_prompt += f"{current_domain.operators[o]}\n"
        codex_prompt += f"{OPERATOR_STOP_TOKEN}\n"

    # Codex prompt for operator definition.
    codex_prompt += f"{OPERATOR_START}{operator_name_to_define}\n"
    if operator_name_to_define in operator_uses:
        for use_example in operator_uses[operator_name_to_define]:
            codex_prompt += f"{EXAMPLE_START}{use_example}\n"
    operator_prefix = f"{OPERATOR_START_TOKEN}{operator_name_to_define}"
    codex_prompt += operator_prefix

    completions = get_completions(
        codex_prompt,
        temperature=temperature,
        stop=OPERATOR_STOP_TOKEN,
        n_samples=n_samples,
    )
    return [operator_prefix + o for o in completions]
