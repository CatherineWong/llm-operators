"""
pddl_parser.py | Utilities related to PDDL.
"""

import copy
import csv
import re
import os
import json

from contextlib import contextmanager
from collections import defaultdict


class Domain:
    def __init__(
        self,
        pddl_domain=None,
        parent_domain=None,
        domain_name=None,
        requirements=None,
        constants=None,
        types=None,
        predicates=None,
        operators=None,
        functions=None,
    ):
        self.pddl_domain = self.init_pddl_domain(pddl_domain)

        self.parent_domain = parent_domain
        self.domain_name = self.init_domain_name(domain_name)
        self.requirements = self.init_simple_pddl(requirements, "requirements")
        self.constants = self.init_simple_pddl(predicates, "constants")
        self.types = self.init_simple_pddl(types, "types")
        self.predicates = self.init_simple_pddl(predicates, "predicates")
        self.functions = self.init_simple_pddl(functions, "functions")
        self.operators = self.init_operators(operators)  # Evaluated operators.
        self.ground_truth_operators = None
        self.ground_truth_predicates = PDDLParser._parse_domain_predicates(
            self.pddl_domain
        )
        self.ground_truth_constants = PDDLParser._parse_constants(self.constants[len('(:constants'):-1])

        # One or more proposed predicates.
        self.proposed_predicates = []
        self.codex_raw_predicates = []
        # One or more operators that have been proposed by Codex at the current iteration.
        self.proposed_operators = defaultdict(list)  # Operator name -> definitions
        self.codex_raw_operators = defaultdict(list)
        self.operators_to_scores = defaultdict(
            float
        )  # (operator_name, body) -> scalar score.
        # Some operators have had standardized names.
        self.operator_canonicalization = {}

        # Additional object types necessary to prompt codex.
        self.codex_types = ""

    def add_additional_constants(self, additional_constant_string):
        self.ground_truth_constants.update(PDDLParser._parse_constants(additional_constant_string))

    def init_pddl_domain(self, pddl_domain):
        if pddl_domain is not None:
            pddl_domain = PDDLParser._purge_comments(pddl_domain)
        return pddl_domain

    def init_domain_name(self, domain_name):
        if domain_name is not None:
            return domain_name
        elif self.parent_domain is not None:
            return self.parent_domain.domain_name
        elif self.pddl_domain is not None:
            patt = r"\(domain(.*?)\)"
            return re.search(patt, self.pddl_domain).groups()[0].strip()
        else:
            return domain_name

    def init_simple_pddl(self, initial_value, str_keyword):
        if initial_value is not None:
            return initial_value
        elif self.parent_domain is not None:
            return vars(self.parent_domain)[str_keyword]
        elif self.pddl_domain is not None:
            try:
                return PDDLParser._find_labelled_expression(
                    self.pddl_domain, f":{str_keyword}"
                )
            except:
                return ""
        return initial_value

    def init_operators(self, initial_value):
        if initial_value is not None:
            return initial_value
        elif self.parent_domain is not None:
            return copy.deepcopy(
                vars(self.parent_domain)["operators"]
            )  # Don't share the operator object.
        elif self.pddl_domain is not None:
            return PDDLParser._parse_domain_operators(self.pddl_domain)
        return initial_value

    def add_operator(self, operator_name, operator_pddl):
        if operator_name in self.operators:
            if operator_pddl == self.operators[operator_name]:
                return
            operator_name = operator_name + f"_{len(self.operators)}"
            print(f"Warning: operator already exists. Renaming to: {operator_name}")
        self.operators[operator_name] = operator_pddl

    def remove_operator(self, operator_name):
        del self.operators[operator_name]

    def get_operator_body(self, operator_name, proposed_operator_index=0):
        if operator_name in self.operators:
            return self.operators[operator_name]
        elif operator_name in self.proposed_operators:
            return self.proposed_operators[operator_name][proposed_operator_index]
        else:
            return False

    def get_canonical_operator(self, operator_name):
        operators_lower = {
            o.lower(): o
            for o in list(self.operators.keys()) + list(self.proposed_operators.keys())
        }
        operators_upper = {
            o.upper(): o
            for o in list(self.operators.keys()) + list(self.proposed_operators.keys())
        }
        if operator_name in list(self.operators.keys()) + list(
            self.proposed_operators.keys()
        ):
            return operator_name
        elif operator_name in operators_lower:
            return operators_lower[operator_name]
        elif operator_name in operators_upper:
            return operators_upper[operator_name]
        else:
            assert False

    def reset_proposed_operators(self):
        self.proposed_operators = defaultdict(list)

    def init_requirements(self, requirements):
        return PDDLParser._find_labelled_expression(self.pddl_domain, ":requirements")

    def operators_to_string(
        self,
        current_operators,
        ground_truth_operators,
        proposed_operators,
        proposed_operator_index=0,
        separator="\n",
        verbose=True,
    ):
        if ground_truth_operators:
            return separator.join(
                [f"""{s}""" for _, s in self.ground_truth_operators.items()]
            )
        else:
            o = ""
            if current_operators:
                o += separator.join([f"""{s}""" for _, s in self.operators.items()])
            o += "\n"
            o += separator.join(
                [
                    f"{self.proposed_operators[o][proposed_operator_index]}"
                    for o in proposed_operators
                    if o in self.proposed_operators
                    and proposed_operator_index < len(self.proposed_operators[o])
                ]
            )

            return o

    def to_string(
        self,
        current_operators=True,
        ground_truth_operators=False,
        proposed_operators=[],
        show_constants=True,
    ):
        domain_str = f"""
    (define (domain {self.domain_name})
        {self.requirements}
        {self.types}
        {self.constants if show_constants else ''}
        {self.predicates}
        {self.functions}
        {self.operators_to_string(current_operators, ground_truth_operators, proposed_operators)}
    )
                """

        return domain_str

    def domain_definition_to_string(self, codex_prompt=False):
        if codex_prompt:
            return "\n".join(
                [
                    self.requirements,
                    self.codex_types,
                    self.types,
                    self.predicates,
                    self.functions,
                ]
            )
        else:
            return "\n".join(
                [self.requirements, self.types, self.predicates, self.functions]
            )

    def domain_for_goal_prompting(
        self, pddl_problem, include_codex_types: bool = False
    ):
        # pddl_problem is the problem string
        # this is to to return shorter version of to_string with only the requirements and types
        problem_types = (
            PDDLParser._find_labelled_expression(pddl_problem, ":objects")
            .split("\n\n")[0]
            .split("\n")[1:-1]
        )
        domain_types = self.types.split("\n")[1:-1]
        type_list = domain_types + problem_types
        types = "(:types\n" + "\n".join(type_list) + ")"
        return f"""
        (define (domain {self.domain_name})
            {self.predicates}
            {types}
            {self.codex_types if include_codex_types else ""}
                    """


class OtherDomain:
    def __init__(
        self, pddl_domain=None,
    ):
        self.pddl_domain = self.init_pddl_domain(pddl_domain)
        self.types = self.init_simple_pddl("types")
        self.predicates = self.init_simple_pddl("predicates")
        self.domain_name = self.init_domain_name()

    def init_pddl_domain(self, pddl_domain):
        if pddl_domain is not None:
            pddl_domain = PDDLParser._purge_comments(pddl_domain)
        return pddl_domain

    def init_simple_pddl(self, str_keyword):
        try:
            return PDDLParser._find_labelled_expression(
                self.pddl_domain, f":{str_keyword}"
            )
        except:
            return ""

    def init_domain_name(self):
        patt = r"\(domain(.*?)\)"
        return re.search(patt, self.pddl_domain).groups()[0].strip()

    def domain_for_goal_prompting(self, pddl_problem):
        # pddl_problem is the problem string
        # this is to to return shorter version of to_string with only the requirements and types
        problem_types = PDDLParser._find_labelled_expression(
            pddl_problem, ":objects"
        ).split("\n")[1:-1]
        domain_types = self.types.split("\n")[1:-1]
        type_list = domain_types + problem_types
        types = "(:types\n" + "\n".join(type_list) + ")"
        return f"""
        (define (domain {self.domain_name})
            {self.predicates}
            {types}
                    """


def save_gt_and_learned_plans(
    curr_iteration, directory, dataset, gt_plans, solved_plans_pddl, problems
):
    plan_filename = os.path.join(directory, f"{dataset}_plans_it_{curr_iteration}.json")

    output_plans = {}
    for plan_id in solved_plans_pddl:
        output_plans[plan_id] = {
            "problem": problems[plan_id].goal_language,
            "solved_plan": str(solved_plans_pddl[plan_id]),
            "gt_plan": str(gt_plans[plan_id]),
        }
    with open(plan_filename, "w") as f:
        json.dump(output_plans, f)
    return plan_filename


def save_learned_operators(curr_iteration, directory, dataset, train_domain, gt_domain):
    operators_filename = os.path.join(
        directory, f"{dataset}_operators_it_{curr_iteration}.json"
    )

    output_operators = {}
    for operator_name in train_domain.operators:
        output_operators[operator_name] = {
            "operator_name": operator_name,
            "pddl_operator": str(train_domain.operators[operator_name]),
        }
    with open(operators_filename, "w") as f:
        json.dump(output_operators, f)
    return operators_filename


def update_pddl_domain_and_problem(
    pddl_domain, problem_idx, problem_id, problems, verbose, command_args,
):
    """Updates the PDDL domain and PDDL problem based on the motion planner results."""
    # Score if operator succeeds.
    operator_success_score = 1
    operator_failure_score = -20  # Penalize an operator heavily for failing
    # Score if task succeeded.
    task_success_score = 1
    task_failure_score = -1

    any_success = False
    for (goal, plan) in problems[problem_id].evaluated_motion_planner_results:
        motion_plan_result = problems[problem_id].evaluated_motion_planner_results[
            (goal, plan)
        ]
        if motion_plan_result.task_success:
            any_success = True

        for operator_idx, o in enumerate(motion_plan_result.pddl_plan.plan):
            # Score the operator for its own success.
            if (
                motion_plan_result.task_success
                or motion_plan_result.last_failed_operator is None
                or operator_idx < motion_plan_result.last_failed_operator
            ):

                pddl_domain.operators_to_scores[
                    (o[PDDLPlan.PDDL_ACTION], o[PDDLPlan.PDDL_OPERATOR_BODY])
                ] += operator_success_score
            else:
                pddl_domain.operators_to_scores[
                    (o[PDDLPlan.PDDL_ACTION], o[PDDLPlan.PDDL_OPERATOR_BODY])
                ] += operator_failure_score
            # Score the operator for the overall task success.
            if motion_plan_result.task_success:
                pddl_domain.operators_to_scores[
                    (o[PDDLPlan.PDDL_ACTION], o[PDDLPlan.PDDL_OPERATOR_BODY])
                ] += task_success_score
            else:
                pddl_domain.operators_to_scores[
                    (o[PDDLPlan.PDDL_ACTION], o[PDDLPlan.PDDL_OPERATOR_BODY])
                ] += task_failure_score
    if verbose:
        print("Top operators after success are:")
        for (o_name, o_body) in sorted(
            pddl_domain.operators_to_scores,
            key=lambda k: pddl_domain.operators_to_scores[k],
            reverse=True,
        ):
            print(o_name, pddl_domain.operators_to_scores[(o_name, o_body)])

    should_continue_planner_attempts = not any_success
    return should_continue_planner_attempts


def checkpoint_and_reset_plans(
    pddl_domain, problems, curr_iteration, command_args, output_directory
):
    experiment_tag = (
        ""
        if len(command_args.experiment_name) < 1
        else f"{command_args.experiment_name}_"
    )
    # Checkpoint all of the task plans regardless of whether they succeeded.
    output_json = [
        problems[problem_id].get_evaluated_pddl_plan_json() for problem_id in problems
    ]
    output_filepath = f"{experiment_tag}task_plans.json"
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)

    # Checkpoint all of the motion plans regardless of whether they succeeded.
    output_json = [
        problems[problem_id].get_evaluated_motion_plan_json() for problem_id in problems
    ]
    output_filepath = f"{experiment_tag}motion_plans.json"
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)
    # Log the human readable motion planner results to a CSV.
    log_motion_planner_results(problems, command_args, output_directory)

    # Reset the plans.
    for problem_id in problems:
        problems[problem_id].update_solved_motion_plan_results()
        problems[problem_id].reset_evaluated_pddl_plans()
        problems[problem_id].reset_evaluated_motion_planner_results()


def checkpoint_and_reset_operators(
    pddl_domain, curr_iteration, command_args, output_directory
):
    # Set operators with final scores.
    OPERATOR_SCORE_THRESHOLD = 0
    for (o_name, o_body) in pddl_domain.operators_to_scores:
        if pddl_domain.operators_to_scores[(o_name, o_body)] > OPERATOR_SCORE_THRESHOLD:
            pddl_domain.add_operator(operator_name=o_name, operator_pddl=o_body)
    # Clear out the proposed operators.
    pddl_domain.reset_proposed_operators()
    # Log operators.
    log_final_operators(pddl_domain, output_directory, command_args.experiment_name)
    print(
        f"Final operators after iteration {curr_iteration}: {pddl_domain.operators.keys()}"
    )


def log_final_operators(pddl_domain, output_directory, experiment_name):
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"
    output_filepath = f"{experiment_tag}final_operators.json"
    # JSON.
    output_json = {
        o_name: pddl_domain.operators[o_name] for o_name in pddl_domain.operators
    }
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)

    # Human readable CSV.
    output_filepath = f"{experiment_tag}final_operators.csv"

    if output_directory:
        print(
            f"Logging final operators: {os.path.join(output_directory, output_filepath)}"
        )
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            fieldnames = [
                "operator_name",
                "gt_operator",
                "final_operator",
                "",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for operator_name, operator in pddl_domain.operators.items():
                writer.writerow(
                    {
                        "operator_name": operator_name,
                        "gt_operator": pddl_domain.ground_truth_operators[operator_name]
                        if operator_name in pddl_domain.ground_truth_operators
                        else "",
                        "final_operator": operator,
                    }
                )


def update_pddl_domain_from_planner_results(
    pddl_domain,
    problems,
    top_n_operators,
    verbose,
    command_args,
    output_directory,
    dataset_name,
):
    """Updates a pddl domain based on scores from evaluating a task and motion planner. Removes other proposed operators."""

    # Simplest possible objective function: assign EXECUTED + TASK_SUCCESS score for any operator on a motion plan that worked. Take the top n.
    #  TODO: LCW - update this to a reasonable scoring function.
    EXECUTED_SCORE = 1
    TASK_SUCCESS_SCORE = 1
    operator_scores = defaultdict(float)
    for problem_id in problems:
        for goal in problems[problem_id].evaluated_motion_planner_results:
            motion_plan_result = problems[problem_id].evaluated_motion_planner_results[
                goal
            ]
            successful_operator_names = [
                o[PDDLPlan.PDDL_ACTION]
                for o in motion_plan_result.pddl_plan.plan[
                    : motion_plan_result.last_failed_operator
                ]
            ]
            for o in successful_operator_names:
                if o not in pddl_domain.operators:
                    operator_scores[o] += EXECUTED_SCORE
                    task_success_score = (
                        TASK_SUCCESS_SCORE if motion_plan_result.task_success else 0
                    )
                    operator_scores[o] += task_success_score

    log_motion_planner_results(problems, command_args, output_directory)

    # Print the top operators and scores.
    top_operators = sorted(
        list(operator_scores.keys()), key=lambda o: -operator_scores[o]
    )[:top_n_operators]
    if verbose:
        print(
            f"Adding {len(top_operators)} operators to the domain with the following scores:"
        )

    experiment_tag = (
        ""
        if len(command_args.experiment_name) < 1
        else f"{command_args.experiment_name}_"
    )
    output_filepath = f"{experiment_tag}operator_scores.json"
    if output_directory:
        print(
            "Logging operator scores to",
            os.path.join(output_directory, output_filepath),
        )
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(operator_scores, f)

    # Add these operator definitions.
    for o in top_operators:
        if verbose:
            print(f"Adding operator: {o} with score {operator_scores[o]}")
            print(pddl_domain.get_operator_body(o))
            assert o not in pddl_domain.operators
            pddl_domain.operators[o] = pddl_domain.get_operator_body(o)
    # Clear out the proposed operators.
    pddl_domain.reset_proposed_operators()


def log_motion_planner_results(problems, command_args, output_directory):
    experiment_tag = (
        ""
        if len(command_args.experiment_name) < 1
        else f"{command_args.experiment_name}_"
    )
    output_filepath = f"{experiment_tag}motion_planner_results.csv"

    if output_directory:
        print(
            "Logging motion planner results to",
            os.path.join(output_directory, output_filepath),
        )

        with open(os.path.join(output_directory, output_filepath), "w") as f:
            fieldnames = [
                "problem_id",
                "goal",
                "task_success",
                "task_plan",
                "task_plan_success_prefix",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for problem in problems.values():
                for goal, r in problem.evaluated_motion_planner_results.items():
                    task_plan = ", ".join([x["action"] for x in r.pddl_plan.plan])
                    task_plan_success_prefix = ", ".join(
                        [
                            x["action"]
                            for x in r.pddl_plan.plan[: r.last_failed_operator]
                        ]
                    )
                    writer.writerow(
                        {
                            "problem_id": problem.problem_id,
                            "goal": goal,
                            "task_success": r.task_success,
                            "task_plan": task_plan,
                            "task_plan_success_prefix": task_plan_success_prefix,
                        }
                    )


@contextmanager
def ablate_operator(domain, operator_name, contents="[insert]"):
    ablated_operator = domain.operators[operator_name]
    domain.operators[
        operator_name
    ] = f"""(:action {operator_name}
    {contents}
)"""
    yield domain
    domain.operators[operator_name] = ablated_operator
    print("restoring", operator_name)


class PDDLParser:
    @classmethod
    def _purge_comments(self, pddl_str):
        # Purge comments from the given string.
        while True:
            match = re.search(r";(.*)\n", pddl_str)
            if match is None:
                return pddl_str
            start, end = match.start(), match.end()
            pddl_str = pddl_str[:start] + pddl_str[end - 1 :]

    @classmethod
    def _parse_domain_operators(cls, pddl_domain):
        matches = re.finditer(r"\(:action", pddl_domain)
        operators = {}
        for match in matches:
            start_ind = match.start()
            op = cls._find_balanced_expression(pddl_domain, start_ind).strip()
            patt = r"\(:action(.*):parameters(.*):precondition(.*):effect(.*)\)"
            op_match = re.match(patt, op, re.DOTALL)
            op_name, params, preconds, effects = op_match.groups()
            op_name = op_name.strip()
            operators[op_name] = op
        return operators

    @classmethod
    def _parse_domain_predicates(cls, pddl_domain):
        start_ind = re.search(r"\(:predicates", pddl_domain).start()
        predicates = cls._find_balanced_expression(pddl_domain, start_ind)

        predicates = predicates[12:-1].strip()
        predicates = cls._find_all_balanced_expressions(predicates)

        predicate_names = {}
        for pred in predicates:
            pred_object = cls._parse_predicate(pred, neg=False)
            predicate_names[pred_object.name] = pred_object
        return predicate_names
    
    @classmethod
    def _parse_problem_predicates(cls, pddl_problem):
        start_ind = re.search(r"\(:init", pddl_problem).start()
        predicates = cls._find_balanced_expression(pddl_problem, start_ind)

        predicates = predicates[12:-1].strip()
        predicates = cls._find_all_balanced_expressions(predicates)

        predicate_names = {}
        for pred in predicates:
            pred_object = cls._parse_predicate(pred, neg=False)
            predicate_names[pred_object.name] = pred_object
        return predicate_names

    @classmethod
    def _parse_predicate(cls, pred, allow_partial_ground_predicates=False, neg=False):
        if allow_partial_ground_predicates:
            pred = pred.strip()[1:-1].split()
        else:
            pred = pred.strip()[1:-1].split("?")

        pred_name = pred[0].strip()
        # arg_types = [self.types[arg.strip().split("-")[1].strip()]
        #              for arg in pred[1:]]
        arg_types = []
        arg_values = []
        for arg in pred[1:]:
            if " - " in arg:
                arg_value = arg.strip().split("-", 1)[0].strip()
                arg_values.append(arg_value)
                arg_type = arg.strip().split("-", 1)[1].strip()
                arg_types.append(arg_type)
            else:
                arg_values.append(arg.strip())
                arg_types.append("")
        return PDDLPredicate(
            pred_name, len(pred[1:]), arg_types, argument_values=arg_values, neg=neg
        )

    @classmethod
    def _parse_constants(cls, strings):
        strings = strings.split('\n')
        arg_types = []
        arg_values = []

        for string in strings:
            arg = string.strip()
            if string == "":
                continue
            if '(' in arg or ')' in arg:
                continue

            if " - " in arg:
                arg_value = arg.strip().split("-", 1)[0].strip()
                arg_values.append(arg_value)
                arg_type = arg.strip().split("-", 1)[1].strip()
                arg_types.append(arg_type)
            else:
                arg_values.append(arg.strip())
                arg_types.append("")
        return dict(zip(arg_values, arg_types))

    @classmethod
    def _find_labelled_expression(cls, string, label):
        # label like :action
        mat = re.search(r"\(" + label, string)
        if mat is None:
            return ""
        start_ind = mat.start()
        return cls._find_balanced_expression(string, start_ind)

    @staticmethod
    def _find_balanced_expression(string, index):
        """Find balanced expression in string starting from given index.
        """
        assert string[index] == "("
        start_index = index
        balance = 1
        while balance != 0:
            index += 1
            symbol = string[index]
            if symbol == "(":
                balance += 1
            elif symbol == ")":
                balance -= 1
        return string[start_index : index + 1]

    @staticmethod
    def _find_all_balanced_expressions(string):
        """Return a list of all balanced expressions in a string,
        starting from the beginning.
        """
        if not string[0] == "(" and string[-1] == ")":
            import pdb

            pdb.set_trace()
        assert string[0] == "("
        assert string[-1] == ")"
        exprs = []
        index = 0
        start_index = index
        balance = 1
        while index < len(string) - 1:
            index += 1
            if balance == 0:
                exprs.append(string[start_index:index])
                # Jump to next "(".
                while True:
                    if string[index] == "(":
                        break
                    index += 1
                start_index = index
                balance = 1
                continue
            symbol = string[index]
            if symbol == "(":
                balance += 1
            elif symbol == ")":
                balance -= 1
        assert balance == 0
        exprs.append(string[start_index : index + 1])
        return exprs


class PDDLPlan:
    PDDL_ACTION = "action"
    PDDL_ARGUMENTS = "args"
    PDDL_OPERATOR_BODY = "operator_body"
    PDDL_GROUND_PREDICATES = "ground_predicates"
    PDDL_INFINITE_COST = 100000

    def __hash__(self):
        return hash(self.plan_string)

    def __eq__(self, other):
        return self.plan_string == other.plan_string

    def __init__(
        self,
        plan=None,
        plan_string=None,
        overall_plan_cost=PDDL_INFINITE_COST,
        pddl_domain=None,
    ):
        self.plan = plan  # list of dictionaries, where each dict is an action
        self.plan_string = plan_string
        if self.plan is None and self.plan_string is not None:
            self.plan = self.string_to_plan(self.plan_string, pddl_domain=pddl_domain)
        if self.plan_string is None and self.plan is not None:
            self.plan_string = self.plan_to_string(self.plan)

        self.overall_plan_cost = overall_plan_cost

    def plan_to_string(self, plan):
        return "\n".join(
            [
                f"({a[PDDLPlan.PDDL_ACTION]} {' '.join(a[PDDLPlan.PDDL_ARGUMENTS])})"
                for a in self.plan
            ]
        )

    def string_to_plan(self, plan_string, pddl_domain=None):
        action_strings = plan_string.strip().split("\n")
        actions = []
        for a in action_strings:
            assert a.startswith("(") and a.endswith(")")
            tokens = a.strip("()").split(" ")
            assert len(tokens) > 0
            actions.append(
                {PDDLPlan.PDDL_ACTION: tokens[0], PDDLPlan.PDDL_ARGUMENTS: tokens[1:]}
            )
        if pddl_domain is not None:
            # Possibly check that we haven't lowercased the actions.
            for action in actions:
                action[PDDLPlan.PDDL_ACTION] = pddl_domain.get_canonical_operator(
                    action[PDDLPlan.PDDL_ACTION]
                )

                operator_body = pddl_domain.get_operator_body(
                    action[PDDLPlan.PDDL_ACTION]
                )
                if operator_body:
                    action[PDDLPlan.PDDL_OPERATOR_BODY] = operator_body
        return actions

    def to_postcondition_predicates_json(self, pddl_domain, remove_alfred_object_ids):
        """
        :ret: [
            {
                "action": OPERATOR_NAME,
                "ground_postcondition_predicates": [
                    "predicate_name": "isHeated",
                    "arguments": "apple",
                    "isNeg": False
                ]
            }
        ]
        """
        postcondition_predicates_json = []
        for action in self.plan:
            ground_postcondition_predicates = PDDLPlan.get_postcondition_predicates(
                action, pddl_domain, remove_alfred_object_ids=remove_alfred_object_ids,
            )
            postcondition_predicates_json.append(
                {
                    PDDLPlan.PDDL_ACTION: action[PDDLPlan.PDDL_ACTION],
                    PDDLPlan.PDDL_GROUND_PREDICATES: [
                        ground_predicate.to_json()
                        for ground_predicate in ground_postcondition_predicates
                    ],
                }
            )
        return postcondition_predicates_json

    @classmethod
    def get_postcondition_predicates(
        cls, action, pddl_domain, remove_alfred_object_ids=True
    ):
        operator_body = pddl_domain.get_operator_body(action[PDDLPlan.PDDL_ACTION])
        # There's a chance that this is a predefined operator, in which case we need to get it directly.

        parameters, processed_preconds, processed_effects = parse_operator_components(
            operator_body, pddl_domain
        )
        parameters_ordered = sorted(parameters.keys())
        ground_arguments_map = {
            argument: ground
            for (argument, ground) in zip(parameters_ordered, action["args"])
        }
        ground_postcondition_predicates = []
        for lifted_predicate in processed_effects:
            ground_arguments = [
                ground_arguments_map[arg] for arg in lifted_predicate.argument_values
            ]
            if remove_alfred_object_ids:
                ground_arguments = [
                    PDDLPredicate.remove_alfred_object_ids(a) for a in ground_arguments
                ]
            ground_postcondition_predicates.append(
                PDDLPredicate(
                    name=lifted_predicate.name,
                    arguments=lifted_predicate.arguments,
                    arg_types=lifted_predicate.argument_values,
                    neg=lifted_predicate.neg,
                    argument_values=ground_arguments,
                )
            )
        return ground_postcondition_predicates

    def __str__(self):
        return "PDDLPlan[{}]".format(self.plan_string.replace("\n", " "))

    def __repr__(self):
        return self.__str__()


class PDDLPredicate:
    PDDL_PREDICATE_NAME = "predicate_name"
    PDDL_PREDICATE_ARGUMENTS = "arguments"
    PDDL_PREDICATE_IS_NEG = "is_neg"

    def __init__(
        self,
        name,
        arguments,
        arg_types,
        argument_values,
        argument_is_ground=[],
        neg=False,
    ):
        """
        name: name of predicate.
        arguments: number of arguments.
        arg_types: domain types of the arguments.
        argument_values: these are the ARGUMENT names of the values.
        arguments_is_ground: boolean list of ground arguments.
        """
        self.name = name
        self.arguments = arguments
        self.arg_types = arg_types
        self.argument_values = argument_values
        self.neg = neg
        self.argument_is_ground = argument_is_ground
        self.static = False

    def mark_static(self, static = True):
        self.static = static

    def to_json(self):
        return {
            PDDLPredicate.PDDL_PREDICATE_NAME: self.name,
            PDDLPredicate.PDDL_PREDICATE_ARGUMENTS: self.argument_values,
            PDDLPredicate.PDDL_PREDICATE_IS_NEG: self.neg,
        }

    @classmethod
    def remove_alfred_object_ids(cls, argument_value):
        return argument_value.split("_")[0]

    def __str__(self):
        if self.neg:
            return f'(not ({self.name} {" ".join(self.argument_values)}))'
        return f'({self.name} {" ".join(self.argument_values)})'

    def __repr__(self):
        return f"PDDLPredicate[{self.__str__()}]"


class PDDLProblem:
    def __init__(self, ground_truth_pddl_problem_string=None):
        self.ground_truth_pddl_problem_string = ground_truth_pddl_problem_string
        self.ground_truth_goal = self.parse_goal_pddl(
            self.ground_truth_pddl_problem_string
        )
        self.ground_truth_goal_list = self.parse_goal_pddl_list(self.ground_truth_goal)
        self.ground_truth_objects_dict = self.parse_problem_objects_pddl(return_dict=True)

    def get_pddl_string_with_proposed_goal(self, proposed_goal):
        # Replaces the ground truth goal with a proposed goal.
        pddl_string = self.ground_truth_pddl_problem_string.replace(
            self.ground_truth_goal, proposed_goal
        )
        return pddl_string

    def parse_goal_pddl(self, pddl_problem):
        pddl_problem = PDDLParser._purge_comments(pddl_problem)
        return PDDLParser._find_labelled_expression(pddl_problem, ":goal")

    def parse_goal_pddl_list(self, pddl_goal_string):
        goal_conjunction = PDDLParser._find_labelled_expression(pddl_goal_string, "and")
        _, preprocessed_predicates, _ = preprocess_conjunction_predicates(
            goal_conjunction,
            ground_truth_predicates=None,
            ground_truth_constants=None,
            allow_partial_ground_predicates=True,
        )
        return preprocessed_predicates

    def parse_goal_for_prompting(self):
        """
        When prompting Codex for goals, we remove the exists statements for alfred problems to match other domain formats
        """
        goal = self.ground_truth_goal.split("\n")
        new_goal = []
        new_goal.extend([row + "\n" for row in goal if "exists" not in row])
        return "".join(new_goal)

    def parse_object_types_to_list(self, object_types):
        """
        object_types is a list of the string form rows of what is listed inside the objects section in a pddl problem
        returns a list of the objects
        """
        object_list = []
        for row in object_types:
            row = row.strip()
            if row == "":
                continue
            instances = row.split("-")[0]
            instances = instances.split()
            object_list.extend(instances)
        return object_list

    def parse_object_types_to_dict(self, object_types):
        """
        object_types is a list of the string form rows of what is listed inside the objects section in a pddl problem
        returns a list of the objects
        """
        object_list = dict()
        for row in object_types:
            row = row.strip()
            if row == "":
                continue
            instances, t = row.split("-")
            t = t.strip()
            instances = instances.split()
            for instance in instances:
                object_list[instance.strip()] = t
        return object_list

    def parse_problem_objects_pddl(self, return_dict=False):
        """
        This parser returns all the objects in the object section in a PDDL problem
        works on both alfred and other supervision domains problems

        based on the assumption there's one type per row, and the different instances are separated by spaces

        returns a list of the objects in the pddl problem
        """
        pddl_problem = PDDLParser._purge_comments(self.ground_truth_pddl_problem_string)
        object_types = PDDLParser._find_labelled_expression(
            pddl_problem, ":objects"
        ).split("\n")[1:-1]
        if return_dict:
            return self.parse_object_types_to_dict(object_types)
        return self.parse_object_types_to_list(object_types)

    def parse_problem_objects_alfred(self):
        """
        same as parse_problem_objects_pddl(), but works only on alfred because of the weird structure of the problem files.
        returns only the objects, without the location.
        Based on the assumption that the objects and location are separated by an empty row

        returns a list of the objects in the pddl problem, without location objects
        """
        pddl_problem = PDDLParser._purge_comments(self.ground_truth_pddl_problem_string)
        # taking the first bunch of objects bc they are separated from location by \n\n
        object_types = (
            PDDLParser._find_labelled_expression(pddl_problem, ":objects")
            .split("\n\n")[0]
            .split("\n")[1:-1]
        )
        return self.parse_object_types_to_list(object_types)


def preprocess_proposed_plans_operators_goals(
    pddl_domain, problems, verbose=False, output_directory=None, command_args=None
):
    # Preprocess operators for correctness.
    preprocess_operators(
        pddl_domain,
        output_directory=output_directory,
        command_args=command_args,
        verbose=verbose,
    )
    # Preprocess goals for correctness.
    preprocess_goals(
        problems=problems,
        pddl_domain=pddl_domain,
        output_directory=output_directory,
        command_args=command_args,
        verbose=verbose,
    )


def preprocess_goals(
    problems, pddl_domain, output_directory, command_args=None, verbose=False
):
    # Preprocess goals, making the hard assumption that we want goals to be conjunctions of existing predicates only.
    unsolved_problems = [
        problems[p]
        for p in problems
        if len(problems[p].solved_motion_plan_results) < 1
        and not problems[p].should_supervise_pddl
    ]
    output_json = dict()
    if verbose:
        print(
            f"preprocess_goals: preprocessing {len(unsolved_problems)} unsolved problems."
        )
    for problem in unsolved_problems:
        preprocessed_goals = []
        for proposed_goal in problem.proposed_pddl_goals:
            if verbose:
                print("Trying to process...")
                print(proposed_goal)
            success, preprocessed_goal = preprocess_goal(
                proposed_goal, pddl_domain, problem.ground_truth_pddl_problem.ground_truth_objects_dict, use_ground_truth_predicates=True
            )
            if not success:
                print("Failed to preprocess goal.")
                print(proposed_goal)

            if success:
                preprocessed_goals.append(preprocessed_goal)
            if verbose:
                print(f"Preprocessed goal: {preprocessed_goal}")
                print("====")
        problem.codex_raw_goals = problem.proposed_pddl_goals
        problem.proposed_pddl_goals = preprocessed_goals
        output_json[problem.problem_id] = preprocessed_goals

    experiment_name = command_args.experiment_name
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"
    output_filepath = f"{experiment_tag}preprocessed_goals.json"
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)
    log_preprocessed_goals(
        problems, output_directory, command_args.experiment_name, verbose
    )


def preprocess_goal(goal, pddl_domain, object_dict, use_ground_truth_predicates=True):
    """
    Preprocesses a goal. Assumes it must be in the following form, with an exists and a set of conjunctions.
    (:goal
        (exists (?o - object)
        (and
            (objectType ?o TomatoType)
            (isSliced ?o)
        )
    )
)
    """
    # Purge comments.
    preprocessed_goal = PDDLParser._purge_comments(goal)

    # TODO(Jiayuan Mao @ 2023/04/07): add "codex_types".
    all_constants = dict()
    all_constants.update(pddl_domain.ground_truth_constants)
    all_constants.update(object_dict)
    # Extract the conjunction.
    try:
        goal_conjunction = PDDLParser._find_labelled_expression(
            preprocessed_goal, "and"
        )
    except:
        print(f"Failure, could not find goal_conjunction in {preprocessed_goal}.")
        return False, ""
    try:
        (
            parameters,
            preprocessed_predicates,
            structured_predicates,
        ) = preprocess_conjunction_predicates(
            goal_conjunction,
            pddl_domain.ground_truth_predicates,
            all_constants,
            allow_partial_ground_predicates=True,
            debug=True,
        )
    except Exception as e:
        print(
            f"Failure, could not find extract ground truth predicates from conjunction in {goal_conjunction}."
        )
        return False, ""
    if parameters is None:
        print(
            f"Failure, could not find extract ground truth predicates from conjunction in {goal_conjunction}."
        )
        return False, ""

    unground_parameters = sorted([p for p in parameters.items() if p[0].startswith("?")])

    # Conjunction
    predicate_string = "\n\t\t".join(preprocessed_predicates)
    preprocessed_goal = f"(and \n\t\t{predicate_string})"

    # Add exists.
    for (variable_name, variable_type) in unground_parameters:
        preprocessed_goal = (
            f"(exists ({variable_name} - {variable_type})\n{preprocessed_goal})"
        )
    # Add goal.
    preprocessed_goal = f"(:goal\n\t{preprocessed_goal}\n)"
    return True, preprocessed_goal


def log_preprocessed_goals(problems, output_directory, experiment_name, verbose=False):
    # Human readable CSV.
    experiment_name = experiment_name
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"
    output_filepath = f"{experiment_tag}preprocessed_goals.csv"

    if output_directory:
        print(
            f"Logging preprocessed goals: {os.path.join(output_directory, output_filepath)}"
        )
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            fieldnames = [
                "problem",
                "nl_goal",
                "gt_pddl_goal",
                "codex_raw_goals",
                "codex_preprocessed_goal",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for p_id in problems:
                problem = problems[p_id]
                for goal in problem.proposed_pddl_goals:
                    writer.writerow(
                        {
                            "problem": problem.problem_id,
                            "nl_goal": problem.language,
                            "gt_pddl_goal": problem.ground_truth_pddl_problem.ground_truth_goal,
                            "codex_raw_goals": problem.codex_raw_goals,
                            "codex_preprocessed_goal": goal,
                        }
                    )


def preprocess_operators(
    pddl_domain,
    output_directory,
    maximum_operator_arity=4,
    command_args=None,
    verbose=False,
):
    # Preprocess operators, making the hard assumption that we want to operators to be conjunctions of existing predicates only.
    output_json = dict()
    logs = dict()

    if verbose:
        print(
            f"preprocess_operators: preprocessing {len(pddl_domain.proposed_operators)} operators."
        )
    for o in list(pddl_domain.proposed_operators.keys()):
        logs[o] = list()
        preprocessed_operators = []
        for proposed_operator_body in pddl_domain.proposed_operators[o]:
            if verbose:
                print("Trying to process...")
                print(proposed_operator_body)
            success, preprocessed_operator = preprocess_operator(
                o,
                proposed_operator_body,
                pddl_domain,
                maximum_operator_arity=maximum_operator_arity,
                use_ground_truth_predicates=True,
            )
            if success:
                logs[o].append((proposed_operator_body, preprocessed_operator))
                preprocessed_operators.append(preprocessed_operator)
            else:
                logs[o].append((proposed_operator_body, "FAILED"))

        pddl_domain.codex_raw_operators[o] = pddl_domain.proposed_operators[o]
        if len(preprocessed_operators) > 0:
            pddl_domain.proposed_operators[o] = preprocessed_operators
            output_json[o] = pddl_domain.proposed_operators[o]
        else:
            del pddl_domain.proposed_operators[o]
        if verbose:
            print(f"Preprocessed operator: {o}")
            print(f"Processed forms {len(preprocessed_operators)}: ")
            if o in pddl_domain.proposed_operators:
                for operator_body in pddl_domain.proposed_operators[o]:
                    print(operator_body)
            print("====")

    # Write out to an output JSON.
    experiment_name = command_args.experiment_name
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"
    output_filepath = f"{experiment_tag}preprocessed_operators.json"
    if output_directory:
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            json.dump(output_json, f)
    log_preprocessed_operators(
        pddl_domain,
        logs,
        output_directory,
        experiment_name=command_args.experiment_name,
        verbose=verbose,
    )


def log_preprocessed_operators(
    pddl_domain, logs, output_directory, experiment_name, verbose=False
):
    # Human readable CSV.
    experiment_name = experiment_name
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"
    output_filepath = f"{experiment_tag}preprocessed_operators.csv"

    if output_directory:
        print(
            f"Logging preprocessed operators: {os.path.join(output_directory, output_filepath)}"
        )
        with open(os.path.join(output_directory, output_filepath), "w") as f:
            fieldnames = [
                "operator_name",
                "gt_operator",
                "codex_raw_operator",
                "codex_preprocessed_operator",
                "",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for operator_name, operator in logs.items():
                for (raw_operator, preprocessed_operator) in operator:
                    writer.writerow(
                        {
                            "operator_name": operator_name,
                            "gt_operator": pddl_domain.ground_truth_operators[
                                operator_name
                            ]
                            if operator_name in pddl_domain.ground_truth_operators
                            else "",
                            "codex_raw_operator": raw_operator,
                            "codex_preprocessed_operator": preprocessed_operator,
                        }
                    )


def parse_operator_components(operator_body, pddl_domain):
    allow_partial_ground_predicates = pddl_domain.constants != ""
    preprocessed_operator = PDDLParser._purge_comments(operator_body)

    matches = re.finditer(r"\(:action", preprocessed_operator)

    for match in matches:
        start_ind = match.start()
        op = PDDLParser._find_balanced_expression(
            preprocessed_operator, start_ind
        ).strip()
        patt = r"\(:action(.*):parameters(.*):precondition(.*):effect(.*)\)"
        op_match = re.match(patt, op, re.DOTALL)
        if op_match is None:
            return False, ""
        op_name, params, preconds, effects = op_match.groups()
        op_name = op_name.strip()
        (
            precond_parameters,
            processed_preconds,
            precondition_predicates,
        ) = preprocess_conjunction_predicates(
            preconds,
            pddl_domain.ground_truth_predicates,
            pddl_domain.ground_truth_constants,
            allow_partial_ground_predicates=allow_partial_ground_predicates,
        )
        if not precond_parameters:
            return False, ""
        (
            effect_parameters,
            processed_effects,
            effect_predicates,
        ) = preprocess_conjunction_predicates(
            effects,
            pddl_domain.ground_truth_predicates,
            pddl_domain.ground_truth_constants,
            allow_partial_ground_predicates=allow_partial_ground_predicates,
            check_static=True
        )
        if not effect_parameters:
            return False, ""
        precond_parameters.update(effect_parameters)

        if allow_partial_ground_predicates:
            # NB(Jiayuan Mao @ 2021/02/04): drop the '?' for parameters, because when allow_partial_ground_predicates is True,
            #   the parameters will have a leading '?'.

            # parameters = {k[1:]: v for k, v in precond_parameters if k.startswith("?")}

            # NB(Lio W: remove the 'drop the leading?')
            parameters = {k: v for k, v in precond_parameters.items() if k.startswith("?")}
        else:
            parameters = precond_parameters

        return parameters, precondition_predicates, effect_predicates


def preprocess_operator(
    operator_name,
    operator_body,
    pddl_domain,
    maximum_operator_arity=4,
    use_ground_truth_predicates=True,
):
    allow_partial_ground_predicates = pddl_domain.constants != ""
    # Purge comments.
    preprocessed_operator = PDDLParser._purge_comments(operator_body)

    matches = re.finditer(r"\(:action", preprocessed_operator)

    for match in matches:
        start_ind = match.start()
        try:
            op = PDDLParser._find_balanced_expression(
                preprocessed_operator, start_ind
            ).strip()
        except IndexError:
            # NB(Jiayuan Mao @ 2023/02/04): sometimes when the length is too short, Codex may
            # propose only a partial operator, which will cause the parser to fail.
            print(f"Failure, operatror proposal is not valid {preprocessed_operator}.")
            return False, ""

        patt = r"\(:action(.*):parameters(.*):precondition(.*):effect(.*)\)"
        op_match = re.match(patt, op, re.DOTALL)
        if op_match is None:
            return False, ""
        op_name, params, preconds, effects = op_match.groups()
        op_name = op_name.strip()
        precond_parameters, processed_preconds, _ = preprocess_conjunction_predicates(
            preconds,
            pddl_domain.ground_truth_predicates,
            pddl_domain.ground_truth_constants,
            allow_partial_ground_predicates=allow_partial_ground_predicates,
        )
        if not precond_parameters:
            return False, ""
        effect_parameters, processed_effects, _ = preprocess_conjunction_predicates(
            effects,
            pddl_domain.ground_truth_predicates,
            pddl_domain.ground_truth_constants,
            allow_partial_ground_predicates=allow_partial_ground_predicates,
            check_static=True
        )
        if not effect_parameters:
            return False, ""
        precond_parameters.update(effect_parameters)

        if not allow_partial_ground_predicates:
            # NB(Jiayuan Mao @ 2023/02/04): if we don't allow partial ground predicates, the parameters do not contain '?'.
            unground_parameters = [
                f"?{name} - {param_type}" for (name, param_type) in precond_parameters.items()
            ]
        else:
            unground_parameters = [
                f"{name} - {param_type}"
                for (name, param_type) in precond_parameters.items()
                if name.startswith("?")
            ]
        if len(unground_parameters) > maximum_operator_arity:
            return False, ""
        params_string = " ".join(unground_parameters)

        precond_string = "\n\t\t".join(processed_preconds)
        precond_string = f"(and \n\t\t{precond_string}\n\t\t)"
        effect_string = "\n\t\t".join(processed_effects)
        effect_string = f"(and \n\t\t{effect_string}\n\t\t)"

        preprocessed_operator = f"""
(:action {op_name}
        :parameters ({params_string})

        :precondition {precond_string}
        :effect {effect_string}
)
        """.strip()
        return True, preprocessed_operator

        # Construct an operator!
    return False, ""


def preprocess_conjunction_predicates(
    conjunction_predicates,
    ground_truth_predicates,
    ground_truth_constants,
    allow_partial_ground_predicates=False,
    check_static=False,
    debug=False,
):
    """Parse a conjunction of predicates.

    Args:
        conjunction_predicates (str): The conjunction of predicates.
        ground_truth_predicates (Dict[str, PDDLPredicate]): The ground truth predicates.
        ground_truth_constants (Dict[str, str]): The ground truth constants, mapping from name to type.
        allow_partial_ground_predicates (bool): Whether to allow constants.
        check_static (bool): Whether to constraint that the predicates are not static (useful for effect predicates).
        debug (bool): Whether to print debug information.

    Returns:
        Tuple[Dict[str, str], List[str], List[PDDLPredicate]]:
            the parameters: mapping from parameter name to type,
            the processed predicates: a list of strings, each string is a processed predicate,
            the predicate names: a list of predicates, not used in the current implementation.
    """

    if conjunction_predicates.strip() == "()":
        return False, [], []
    patt = r"\(and(.*)\)"
    op_match = re.match(patt, conjunction_predicates.strip(), re.DOTALL)

    if not op_match:
        return None, None, None

    if len(op_match.groups()) != 1 and debug:
        import pdb

        pdb.set_trace()

    parameters = dict()
    conjunction_predicates = op_match.groups()[0].strip()
    if len(conjunction_predicates) <= 0:
        return None, None, None
    predicates_list = [
        p.strip()
        for p in PDDLParser._find_all_balanced_expressions(op_match.groups()[0].strip())
    ]
    preprocessed_predicates = []
    structured_predicates = []
    # print("predicates_list", predicates_list)
    for pred_string in predicates_list:
        patt = r"\(not(.*)\)"
        not_match = re.match(patt, pred_string, re.DOTALL)
        if not_match is not None:
            neg = True
            inner_predicate = not_match.groups()[0].strip()
        else:
            neg = False
            inner_predicate = pred_string

        parsed_predicate = PDDLParser._parse_predicate(
            inner_predicate,
            allow_partial_ground_predicates=allow_partial_ground_predicates,
            neg=neg,
        )

        if ground_truth_predicates is not None and (
            (parsed_predicate.name not in ground_truth_predicates)
            or parsed_predicate.arguments != ground_truth_predicates[parsed_predicate.name].arguments
        ):
            continue
        else:
            valid = True
            if ground_truth_predicates is not None:
                if check_static and ground_truth_predicates[parsed_predicate.name].static:
                    continue

                # NB(Jiayuan Mao @ 2023/04/07): if the new predicate is not valid, we restore the original parameters.
                parameters_backup = parameters.copy()
                for argname, argtype in zip(
                    parsed_predicate.argument_values,
                    ground_truth_predicates[parsed_predicate.name].arg_types,
                ):
                    if argname.startswith('?'):
                        if argname not in parameters:
                            parameters[argname] = argtype
                        else:
                            if parameters[argname] != argtype:
                                valid = False
                                break
                    else:
                        if ground_truth_constants is not None:
                            if argname not in ground_truth_constants or ground_truth_constants[argname] != argtype:
                                valid = False
                                break
            if valid:
                preprocessed_predicates.append(pred_string)
                structured_predicates.append(parsed_predicate)
            else:
                parameters = parameters_backup

    return parameters, preprocessed_predicates, structured_predicates

