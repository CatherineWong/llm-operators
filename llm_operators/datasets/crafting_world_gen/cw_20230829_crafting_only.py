#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as npr
from typing import Any, Optional, Tuple, Dict

from llm_operators.datasets.dataset_core import Problem
from llm_operators.datasets.crafting_world_gen.crafting_world_rules import get_all_crafting_ingradients, get_all_crafting_locations, get_all_crafting_outcomes, CRAFTING_RULES
from llm_operators.datasets.crafting_world_gen.utils import underline_to_pascal, underline_to_space
from llm_operators.pddl import PDDLProblem, PDDLPlan


PROBLEM_PDDL_TEMPLATE = """
(define
 (problem crafting-world-v20230106-p{problem_id})
 (:domain crafting-world-v20230106)
 (:objects
   {objects_str}
 )
 (:init
   {init_str}
 )
 (:goal (and
   {goal_str}
 ))
)
"""


def gen_linear_tile(n: int) -> Tuple[str, str]:
    """Generate the object string and init string for a linear 1D map.

    .. code::

        t1 t2 t3 t4 ... tn

    Args:
        n: the length of the map.

    Returns:
        a tuple of the object string and init string.
    """

    object_str = ""
    init_str = ""
    for i in range(1, n + 1):
        object_str += f"t{i} - tile\n"
        if i > 1:
            init_str += f"(tile-right t{i - 1} t{i})\n"
            init_str += f"(tile-left t{i} t{i - 1})\n"
    return object_str, init_str


def gen_locations_and_objects(n: int, inventory_size: int) -> Tuple[str, str, int, list, list]:
    """Sample a linear map by randomly placing mining locations and tools on the map."""

    crafting_locations = get_all_crafting_locations()
    crafting_ingredients = get_all_crafting_ingradients()

    nr_objects = len(crafting_locations) + len(crafting_ingredients) + inventory_size

    map_location = [None for _ in range(n)]
    map_objects = [[] for _ in range(n)]

    assert len(crafting_locations) < n, 'Map is too small to fit all crafting locations: {} < {}'.format(n, len(crafting_locations))

    object_str = ''
    init_str = ''

    crafting_location_assignments = npr.choice(np.arange(n), len(crafting_locations), replace=False)
    for i, x in enumerate(crafting_location_assignments):
        map_location[x] = (i + 1, crafting_locations[i])
        object_str += f'o{i + 1} - object\n'
        init_str += '(object-of-type o{} {})\n'.format(i + 1, underline_to_pascal(crafting_locations[i]))
        init_str += '(object-at o{} t{})\n'.format(i + 1, x + 1)

    crafting_ingradients_assignments = npr.choice(np.arange(n), len(crafting_ingredients), replace=True)
    for i, x in enumerate(crafting_ingradients_assignments):
        map_objects[x].append((i + 1 + len(crafting_locations), crafting_ingredients[i]))
        object_str += f'o{i + 1 + len(crafting_locations)} - object\n'
        init_str += '(object-of-type o{} {})\n'.format(i + 1 + len(crafting_locations), underline_to_pascal(crafting_ingredients[i]))
        init_str += '(object-at o{} t{})\n'.format(i + 1 + len(crafting_locations), x + 1)

    for i in range(inventory_size):
        object_str += f'o{i + 1 + len(crafting_locations) + len(crafting_ingredients)} - object\n'
        object_str += f'i{i + 1} - inventory\n'
        init_str += '(object-of-type o{} Hypothetical)\n'.format(i + 1 + len(crafting_locations) + len(crafting_ingredients))
        init_str += '(inventory-empty i{})\n'.format(i + 1)

    return object_str, init_str, nr_objects, map_location, map_objects


def gen_v20230829_instance_record(problem_id: str, split: str, n: int = 15, inventory_size: int = 3) -> Dict[str, Any]:
    import jacinle

    crafting_outcomes = get_all_crafting_outcomes()
    goal = npr.choice(crafting_outcomes)

    obj_object_str, obj_init_str, nr_objects, map_location, map_objects = gen_locations_and_objects(n, inventory_size)
    map_object_str, map_init_str = gen_linear_tile(n)
    object_str = obj_object_str + map_object_str
    init_str = obj_init_str + map_init_str
    init_str += '(agent-at t1)\n'

    goal_nl = 'Craft a {}.'.format(underline_to_space(goal))
    goal_str = f'(inventory-holding i{inventory_size} o{nr_objects})\n(object-of-type o{nr_objects} {underline_to_pascal(goal)})\n'

    problem_pddl = PROBLEM_PDDL_TEMPLATE.format(
        problem_id=problem_id,
        objects_str=jacinle.indent_text(object_str, indent_format='   ').strip(),
        init_str=jacinle.indent_text(init_str, indent_format='   ').strip(),
        goal_str=jacinle.indent_text(goal_str, indent_format='   ').strip(),
    )

    return dict(
        problem_id=problem_id,
        split=split,
        problem_pddl=problem_pddl,
        goal_nl=goal_nl,
        goal=goal,
        map_location=map_location,
        map_objects=map_objects,
        nr_objects=nr_objects,
        inventory_size=inventory_size,
    )


def gen_v20230829_solution(record: Dict[str, Any]) -> Optional[PDDLPlan]:
    """Solve the problem with a human-written solution.

    .. warning::

        This is a human-written solution, and is not guaranteed to be optimal.
        Also, this solution only considers the first rule that match the goal. Therefore, it may say "no plan" when there is a plan.

    Args:
        record: the instance record, returned by :meth:`gen_v20230829_instance_record`.

    Returns:
        a PDDLPlan object.
    """

    target_rule = None
    for rule in CRAFTING_RULES:
        if rule['create'] == record['goal']:
            target_rule = rule
            break

    if target_rule is None:
        return None

    current_pos = 1
    actions = list()
    tool_object_ids = list()
    for i, x in enumerate(target_rule['recipe']):
        location, tool_object_id = _find_object_on_the_map(record, x)
        actions.append(f'(move-to t{current_pos} t{location + 1})')
        actions.append(f'(pick-up i{i + 1} o{tool_object_id} t{location + 1})')
        tool_object_ids.append(tool_object_id)
        current_pos = location + 1

    location, loc_object_id = _find_location_on_the_map(record, target_rule['location'])
    assert location is not None
    actions.append(f'(move-to t{current_pos} t{location + 1})')

    if len(target_rule['recipe']) == 1:
        action_name = target_rule['rule_name'].replace('_', '-')
        inventory_str = 'i' + str(record['inventory_size'])
        object_str = 'o' + str(record['nr_objects'])
        actions.append(f'({action_name} i1 {inventory_str} o{loc_object_id} o{tool_object_ids[0]} {object_str} t{location + 1})')
    elif len(target_rule['recipe']) == 2:
        action_name = target_rule['rule_name'].replace('_', '-')
        inventory_str = 'i' + str(record['inventory_size'])
        object_str = 'o' + str(record['nr_objects'])
        actions.append(f'({action_name} i1 i2 {inventory_str} o{loc_object_id} o{tool_object_ids[0]} o{tool_object_ids[1]} {object_str} t{location + 1})')

    return PDDLPlan(plan_string='\n'.join(actions))


def _find_location_on_the_map(record: Dict[str, Any], obj: str) -> Tuple[Optional[int], Optional[int]]:
    for i, x in enumerate(record['map_location']):
        if x is not None and x[1] == obj:
            return i, x[0]
    return None, None


def _find_object_on_the_map(record: Dict[str, Any], obj: str) -> Tuple[Optional[int], Optional[int]]:
    for i, x in enumerate(record['map_objects']):
        for j, y in enumerate(x):
            if y[1] == obj:
                return i, y[0]
    return None, None


def problem_from_raw_record(record: Dict[str, Any]) -> Problem:
    return Problem(
        problem_id=record['problem_id'],
        dataset_split=record['split'],
        language=record['goal_nl'],
        ground_truth_pddl_plan=gen_v20230829_solution(record),
        ground_truth_pddl_problem=PDDLProblem(record['problem_pddl']),
        goal_prefix=f'mining_{record["goal"]}',
    )
