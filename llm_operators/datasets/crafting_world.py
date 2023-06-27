#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
import itertools
import json

from datasets.core import register_planning_pddl_domain, register_planning_domain_problems
from datasets.dataset_utils import load_pddl_file_with_operators
from datasets.crafting_world_gen.cw_20230204_minining_only import problem_from_raw_record, gen_v20230204_solution
from datasets.crafting_world_gen.utils import underline_to_pascal
from datasets.crafting_world_gen.crafting_world_rules import MINING_RULES

CRAFTING_WORLD_PDDL_DOMAIN_NAME = 'crafting_world'
CRAFTING_WORLD_PDDL_DOMAIN_FILE = 'data/domains/crafting_world/domain.pddl'

CRAFTING_WORLD_TELEPORT_DOMAIN_NAME = 'crafting_world_teleport'
CRAFTING_WORLD_TELEPORT_DOMAIN_FILE = 'data/domains/crafting_world_teleport/domain.pddl'

CRAFTING_WORLD_STATIC_PREDICATES = [
    'tile-up', 'tile-down', 'tile-left', 'tile-right',
]


@register_planning_pddl_domain(CRAFTING_WORLD_PDDL_DOMAIN_NAME)
def load_crafting_world_pddl_domain(verbose=False):
    domain = load_pddl_file_with_operators(
        domain_name=CRAFTING_WORLD_PDDL_DOMAIN_NAME,
        file_path=CRAFTING_WORLD_PDDL_DOMAIN_FILE,
        verbose=verbose,
    )
    for predicate in CRAFTING_WORLD_STATIC_PREDICATES:
        domain.ground_truth_predicates[predicate].mark_static()
    domain.codex_types = CRAFTING_WORLD_CODEX_TYPES
    return domain


@register_planning_pddl_domain(CRAFTING_WORLD_TELEPORT_DOMAIN_NAME)
def load_crafting_world_teleport_pddl_domain(verbose=False):
    domain = load_pddl_file_with_operators(
        domain_name=CRAFTING_WORLD_TELEPORT_DOMAIN_NAME,
        file_path=CRAFTING_WORLD_TELEPORT_DOMAIN_FILE,
        verbose=verbose,
    )
    for predicate in CRAFTING_WORLD_STATIC_PREDICATES:
        domain.ground_truth_predicates[predicate].mark_static()
    domain.codex_types = CRAFTING_WORLD_CODEX_TYPES
    return domain


CRAFTING_WORLD_20230204_DATASET_NAME = 'crafting_world_20230204_minining_only'

@register_planning_domain_problems(CRAFTING_WORLD_20230204_DATASET_NAME)
def load_crafting_world_20230204_minining_only(dataset_pddl_directory: str, dataset_fraction: float, verbose=False):
    """Experiments crafting_world_v20230204_mining_only

        1. The map is a linear chain: t1 t2 t3 ...
        2. Agent only needs to accomplish mining tasks in this environment.
        3. All types of resources (e.g., Tree, IronOre) will be randomly distributed on the map.
        4. All types of tools (only tools that can be used for mining) will be randomly distributed on the map.
        5. The agent starts at tile 1.
        6. To solve the task, the "GT" plan is to find a corresponding tool, pick it up, navigate to the resource tile, and execute mining.
        7. Instructions are generated with a single template: Mine X from the map.

    .. code::
        python main.py --experiment_name cw_v20230204_mining_only_full --dataset_name crafting_world_20230204_minining_only --supervision_name supervision --pddl_domain_name crafting_world --dataset_fraction 1.0 --training_plans_fraction 1.0 --initial_plans_prefix mining --initial_pddl_operators move-up move-down move-left move-right pick-up place-down mine-iron-ore --verbose --train_iterations 1 --dataset_pddl_directory data/dataset/crafting_world_v202302024_mining_only --goal_propose_include_codex_types --operator_propose_minimum_usage 1 --output_directory generated --debug_stop_after_first_proposal

    Note that since the task planner will timeout on this domain, we need to implement a new task planner (skipped for now, only testing for proposals).
    See generated/cw_v20230204_mining_only_full/0/cw_v20230204_mining_only_full_preprocessed_operators.csv for results.
    """
    with open(osp.join(dataset_pddl_directory, 'dataset.json')) as f:
        dataset = json.load(f)

    for split, split_problems in dataset.items():
        dataset[split] = {
            problem['problem_id']: problem_from_raw_record(problem)
            for problem in split_problems[:int(len(split_problems) * dataset_fraction)]
        }

    assert len(dataset['train']) > 3
    for problem in itertools.islice(dataset['train'].values(), 3):
        problem.should_supervise_pddl = True

    return dataset


class CraftingWorld20230204Simulator(object):
    def __init__(self):
        self.nr_grids = 15
        self.agent_pos = 1
        self.objects = dict()  # str: (str, int), name: (type, pos)
        self.inventory = dict()  # int: Optional[Tuple[str, str]]  # (type, name)
        self.hypothetical = set()  # str

    def reset_from_state(self, objects, state):
        agent_at = list(state['agent-at'])[0][0]
        self.agent_pos = int(agent_at[1:])

        nr_inventory = len(objects['inventory'])

        self.objects = dict()
        self.inventory = {i: None for i in range(1, 1 + nr_inventory)}

        for obj_name, obj_loc in state.get('object-at', []):
            obj_type = None
            for obj_name2, obj_type2 in state['object-of-type']:
                if obj_name2 == obj_name:
                    obj_type = obj_type2
                    break
            assert obj_type is not None
            self.objects[obj_name] = (obj_type, int(obj_loc[1:]))

        for inv_id, obj_name in state.get('inventory-holding', []):
            obj_type = None
            for obj_name2, obj_type2 in state['object-of-type']:
                if obj_name2 == obj_name:
                    obj_type = obj_type2
            assert obj_type is not None
            self.inventory[int(inv_id[1:])] = (obj_type, obj_name)

        for obj_name, obj_type in state['object-of-type']:
            if obj_type == 'Hypothetical':
                self.hypothetical.add(obj_name)

    def move_to(self, pos):
        self.agent_pos = max(1, min(self.nr_grids, pos))

    def move_left(self):
        self.agent_pos = max(1, self.agent_pos - 1)
        return True

    def move_right(self):
        self.agent_pos = min(self.nr_grids, self.agent_pos + 1)
        return True

    def pick_up(self, inventory, obj_name):
        if self.inventory[inventory] is not None:
            return False
        if self.objects[obj_name][1] != self.agent_pos:
            return False

        self.inventory[inventory] = self.objects[obj_name][0], obj_name
        del self.objects[obj_name]
        return True

    def place_down(self, inventory):
        if self.inventory[inventory] is None:
            return False

        obj_type, obj_name = self.inventory[inventory]
        self.objects[obj_name] = obj_type, self.agent_pos

    def mine(self, obj_name, inventory, hypothetical_object_name, tool_inventory=None):
        if self.objects[obj_name][1] != self.agent_pos:
            return False
        if self.inventory[inventory] is not None:
            return False
        if hypothetical_object_name not in self.hypothetical:
            return False
        if tool_inventory is not None and self.inventory[tool_inventory] is None:
            return False

        obj_type, _ = self.objects[obj_name]

        for rule in MINING_RULES:
            if underline_to_pascal(rule['location']) == obj_type:
                if tool_inventory is None:
                    if len(rule['holding']) == 0:
                        new_obj_type = underline_to_pascal(rule['create'])
                        self.inventory[inventory] = (new_obj_type, hypothetical_object_name)
                        self.hypothetical.remove(hypothetical_object_name)
                        return True
                else:
                    tool_type, _ = self.inventory[tool_inventory]
                    if len(rule['holding']) == 0 or (len(rule['holding']) == 1 and underline_to_pascal(rule['holding'][0]) == tool_type):
                        new_obj_type = underline_to_pascal(rule['create'])
                        self.inventory[inventory] = (new_obj_type, hypothetical_object_name)
                        self.hypothetical.remove(hypothetical_object_name)
                        return True

        return False

    def goal_satisfied(self, goals):
        for goal in goals:
            parts = goal.split(' ')
            if parts[0] == 'inventory-holding':
                inv_id = int(parts[1][1:])
                obj_name = parts[2]
                if self.inventory[inv_id] is None:
                    return False
                if self.inventory[inv_id][1] != obj_name:
                    return False
            elif parts[0] == 'object-of-type':
                obj_name = parts[1]
                obj_type = parts[2]

                found = False
                if obj_name in self.objects:
                    found = True
                    if self.objects[obj_name][0] != obj_type:
                        return False

                if found:
                    continue

                for inv in self.inventory.values():
                    if inv is not None:
                        obj_type2, obj_name2 = inv
                        print(obj_name2, obj_type2, obj_name, obj_type)
                        if obj_name == obj_name2 and obj_type == obj_type2:
                            found = True
                            break

                if not found:
                    return False
            else:
                raise NotImplementedError()

        return True


CRAFTING_WORLD_CODEX_TYPES = """
 (:constants
  Key - object-type
  WorkStation - object-type
  Pickaxe - object-type
  IronOreVein - object-type
  IronOre - object-type
  IronIngot - object-type
  CoalOreVein - object-type
  Coal - object-type
  GoldOreVein - object-type
  GoldOre - object-type
  GoldIngot - object-type
  CobblestoneStash - object-type
  Cobblestone - object-type
  Axe - object-type
  Tree - object-type
  Wood - object-type
  WoodPlank - object-type
  Stick - object-type
  WeaponStation - object-type
  Sword - object-type
  Chicken - object-type
  Feather - object-type
  Arrow - object-type
  ToolStation - object-type
  Shears - object-type
  Sheep - object-type
  Wool - object-type
  Bed - object-type
  BedStation - object-type
  BoatStation - object-type
  Boat - object-type
  SugarCanePlant - object-type
  SugarCane - object-type
  Paper - object-type
  Furnace - object-type
  FoodStation - object-type
  Bowl - object-type
  PotatoPlant - object-type
  Potato - object-type
  CookedPotato - object-type
  BeetrootCrop - object-type
  Beetroot - object-type
  BeetrootSoup - object-type

  Hypothetical - object-type
  Trash - object-type
 )
"""
