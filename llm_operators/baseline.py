"""
baseline.py
Call LLM, condition on a prompt with natural language goal, goal, entire problem specification, 
and list of predicates. 
Output predicate list (instead of task planner) - list of postconditions that should be true.
"""

import csv
import os
import random
import time
import json
import codex
import pddl
import datasets.core
from collections import Counter, defaultdict

import openai
from openai.error import APIConnectionError, InvalidRequestError, RateLimitError

from llm_operators.pddl import PDDLPlan

NONE = "NONE"
STOP_TOKEN = "\n<END>\n"
OPERATOR_START = ";; Operator: "
NATURAL_LANGUAGE_GOAL_START = ";; Goal: "
PDDL_GOAL_START = ";; PDDL Goal: "
PRECONDITION_START = ";; Pre-Conditions: "
POSTCONDITION_START = ";; Post-Conditions: "
BASELINE_TITLE = ";; Given the PDDL postcondition predicates, for each natural language goal,PDDL goal, and PDDL initial conditions, come up with a series of PDDL postconditions that should be satisfied, in order, to satisfy the goal."

DEFAULT_GOAL_TEMPERATURE = 0.0

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is not set. Please set this in the shell via `export OPENAI_API_KEY=...`"
    )
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_problem_prompt(problem : datasets.core.Problem, solved = False):
    nl_goal = NATURAL_LANGUAGE_GOAL_START + problem.ground_truth_goal + "\n"
    pddl_goal = PDDL_GOAL_START + problem.parse_goal_for_prompting() + "\n"
    # TODO get the predicated out of a problem
    # preconditions = PRECONDITION_START + problem.ground_truth_pddl_problem.get_preconditions_predicates()
    problem_prompt = nl_goal + pddl_goal + preconditions
    # TODO @Lio - implement get_postconditions()
    # if solved:
    #     problem_prompt += POSTCONDITION_START + "\n".join(problem.ground_truth_pddl_problem.get_postconditions())


def baseline_prompt(domain: pddl.Domain,problem : pddl.PDDLProblem,n_examples = 1):
    """
    @param domain the domain we're working with
    @param problem the problem we want to prompt with
    @param n_examples the number of examples to be appended to the prompt before getting completions
    @returns a prompt to send to the model
    """
    nl_header = BASELINE_TITLE
    predicates = ";;;; Predicates in the PDDL domain definition.\n" + "\n".join(list(domain.ground_truth_predicates.values))
    codex_prompt = (nl_header + 
            + predicates
            + "\n\n")
    # TODO: randomly select problems from the archive
    # for i in range(n_examples):
    #     codex_prompt += get_problem_prompt(problem, solved = True)
    return codex_prompt

def get_gpt4_completions(domain,output_path,n_completions = 1):

    for i in range(n_completions):
    # TODO: randomly select problems from the archive
        # problem = ... 
        prompt = baseline_prompt(domain,problem)
        completion = codex.get_completions(prompt = prompt,engine = "gpt-4")

    # TODO parse into a json file