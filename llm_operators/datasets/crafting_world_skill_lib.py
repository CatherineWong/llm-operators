#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from llm_operators.datasets.crafting_world import CraftingWorld20230204Simulator


def move_to_resource(s: CraftingWorld20230204Simulator, resource_name: str):
    """Move to the nearest resource."""

    for o, (t, x) in s.objects.items():
        if t == resource_name:
            s.move_to(x)
            return o


def find_object(s: CraftingWorld20230204Simulator, object_type_name: str):
    """Find the nearest object of the target type"""

    for o, (t, x) in s.objects.items():
        if t == object_type_name:
            return o


def pickup_object(s: CraftingWorld20230204Simulator, inventory: int, object_name: str):
    """Pick up the object."""
    assert object_name in s.objects
    rv = s.move_to(s.objects[object_name][1])
    if not rv:
        return False
    rv = s.pick_up(inventory, object_name)
    if not rv:
        return False
    return True


def find_empty_inventory(s: CraftingWorld20230204Simulator, other_than: set[int]):
    """Find a empty inventory."""
    for i, content in s.inventory.items():
        if content is None:
            if i not in other_than:
                return i


def find_hypothetical_object(s: CraftingWorld20230204Simulator, other_than: set[str]):
    """Find a hypothetical object"""

    for x in list(s.hypothetical):
        if x not in other_than:
            return x


def get_inventory_by_object_type(s: CraftingWorld20230204Simulator, object_type: str):
    """Get the object in the inventory."""

    for i, content in s.inventory.items():
        if content is None:
            continue
        t, object_id = content
        if t == object_type:
            return i


def mine_wood(s: CraftingWorld20230204Simulator, inventory: int, target_object: str):
    """Mine wood."""

    tool_inventory = get_inventory_by_object_type(s, 'Axe')
    if tool_inventory is None:
        object_id = find_object(s, 'Axe')
        if object_id is None:
            return False
        tool_inventory = find_empty_inventory(s, {inventory})
        rv = pickup_object(s, tool_inventory, object_id)
        if not rv:
            return False

    tree_object = move_to_resource(s, 'Tree')
    if tree_object is None:
        return False
    rv = s.mine(tree_object, inventory, target_object, tool_inventory)
    if not rv:
        return False
    return True


def mine_potato(s: CraftingWorld20230204Simulator, inventory: int, target_object: str):
    """Mine potato."""

    potato_plant_object = move_to_resource(s, 'PotatoPlant')
    if potato_plant_object is None:
        return False
    rv = s.mine(potato_plant_object, inventory, target_object)
    if not rv:
        return False
    return True

