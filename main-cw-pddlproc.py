#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import jacinle

import llm_operators.pddl as pddl
import llm_operators.datasets as datasets

import concepts.dsl.expression as E
import concepts.pdsketch as pds
from concepts.benchmark.gridworld.crafting_world.crafting_world_gen.cw_20230913_mixed import gen_v20230913_instance_record

parser = jacinle.JacArgumentParser()
parser.add_argument('--mode', default='load', choices=['load', 'gt'])
parser.add_argument('--pddl_domain_name', type=str)
parser.add_argument("--initial_pddl_operators", type=str, nargs="+", help="Which initial PDDL operators to run with.  Used to seed the Codex proposals.")
parser.add_argument('--pddl_domain_file', type=str)
parser.add_argument("--operator_pseudocounts", type=int, default=0.1, help="Assume each operator succeeded at least this many times (MAP smoothing)")
# parser.add_argument('--directory', required=True)
args = parser.parse_args()


CW_BASE_REGRESSION_RULES = r"""
 (:regression move [always]
  :parameters ((forall ?t1 - tile) ?t2 - tile)
  :goal (agent-at ?t2)
  :precondition (and (agent-at ?t1))
  :rule (then
    (move-to ?t1 ?t2)
  )
 )
 (:regression pick-up [always]
  :parameters (?target-type - object-type (forall ?target-inventory - inventory) (forall ?target - object) (forall ?t - tile))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x ?target-type))))
  :precondition (and (object-at ?target ?t) (object-of-type ?target ?target-type) (inventory-empty ?target-inventory))
  :rule (then
    (achieve (agent-at ?t))
    (pick-up ?target-inventory ?target ?t)
  )
 )
"""

CW_REGRESSION_RULE_0 = r"""
 (:regression {action_name}-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type}))))
  :precondition (and (object-at ?target-resource ?t) (object-of-type ?target-resource {station_type}) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical))
  :rule (then
    (achieve (agent-at ?t))
    ({action_name} {param_string})
  )
 )
"""

CW_REGRESSION_RULE_1 = r"""
 (:regression {action_name}-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ing1 - object) (forall ?ing1-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type}))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource {station_type}) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ing1-inventory ?ing1) (object-of-type ?ing1 {ing1_type})
  )
  :rule (then
    (achieve (agent-at ?t))
    ({action_name} {param_string})
  )
 )
 (:regression {action_name}-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type}))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {ing1_type})))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type})))))
  )
 )
"""

CW_REGRESSION_RULE_2 = r"""
 (:regression {rule_name}-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ing1 - object) (forall ?ing1-inventory - inventory) (forall ?ing2 - object) (forall ?ing2-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type}))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource {station_type}) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ing1-inventory ?ing1) (object-of-type ?ing1 {ing1_type})
    (inventory-holding ?ing2-inventory ?ing2) (object-of-type ?ing2 {ing2_type})
  )
  :rule (then
    (achieve (agent-at ?t))
    ({action_name} {param_string})
  )
 )
 (:regression {rule_name}-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type}))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {ing1_type})))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {ing2_type})))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type})))))
  )
 )
 """


def main():
    if args.mode == 'gt':
        assert args.pddl_domain_name is not None
        pddl_domain = datasets.load_pddl_domain(args.pddl_domain_name, args.initial_pddl_operators, verbose=True)
        pddl_domain.init_operators_to_scores(args.operator_pseudocounts)
        domain_string = pddl_domain.to_string()
        domain = pds.load_domain_string(domain_string)
    elif args.mode == 'load':
        assert args.pddl_domain_file is not None
        domain = pds.load_domain_file(args.pddl_domain_file)
    else:
        raise ValueError('Unknown mode: {}'.format(args.mode))

    # from concepts.benchmark.gridworld.crafting_world.crafting_world_teleport import get_domain_filename
    # ref_domain = pds.load_domain_file(get_domain_filename())
    _patch_cw_regression_rules(domain)

    executor = pds.PDSketchExecutor(domain)
    problem_func = gen_v20230913_instance_record

    nr_total = 0
    nr_succ = 0
    for i in range(23):
        record = problem_func(f'test-{i}', 'train', goal_index=i)
        succ = plan(executor, record)
        nr_total += 1
        nr_succ += int(succ)

    print(f'Success rate: {nr_succ} / {nr_total} = {nr_succ / nr_total}')


def plan(executor, record):
    problem_pddl = record['problem_pddl']
    initial_state, goal = pds.load_problem_string(problem_pddl, executor.domain, executor=executor)

    # print(initial_state)
    # print(goal)

    start_time = time.time()
    from concepts.pdsketch.planners.optimistic_search_bilevel_legacy import enumerate_possible_symbolic_plans_regression_c_v2
    rv, stat = enumerate_possible_symbolic_plans_regression_c_v2(executor, initial_state, goal, enable_reordering=False, verbose=False, max_depth=10)

    if len(rv) == 0:
        print(f'!!!No plan found for goal {goal}.')
        return False

    table = list()
    plan = rv[0][0]
    table.append(('goal', str(goal)))
    table.append(('plan_length', len(plan)))
    table.append(('plan', '\n'.join([str(a) for a in plan])))
    table.append(('time', f'{time.time() - start_time:.3f}s'))
    table.extend(stat.items())
    print(jacinle.tabulate(table, tablefmt='simple'))
    return True


CW_BASE_REGRESSION_RULES = r"""
 (:regression move [always]
  :parameters ((forall ?t1 - tile) ?t2 - tile)
  :goal (agent-at ?t2)
  :precondition (and (agent-at ?t1))
  :rule (then
    (move-to ?t1 ?t2)
  )
 )
 (:regression pick-up [always]
  :parameters (?target-type - object-type (forall ?target-inventory - inventory) (forall ?target - object) (forall ?t - tile))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x ?target-type))))
  :precondition (and (object-at ?target ?t) (object-of-type ?target ?target-type) (inventory-empty ?target-inventory))
  :rule (then
    (achieve (agent-at ?t))
    (pick-up ?target-inventory ?target ?t)
  )
 )
"""

CW_REGRESSION_RULE_0 = r"""
 (:regression {action_name}-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type}))))
  :precondition (and (object-at ?target-resource ?t) (object-of-type ?target-resource {station_type}) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical))
  :rule (then
    (achieve (agent-at ?t))
    ({action_name} {param_string})
  )
 )
"""

CW_REGRESSION_RULE_1 = r"""
 (:regression {action_name}-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ing1 - object) (forall ?ing1-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type}))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource {station_type}) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ing1-inventory ?ing1) (object-of-type ?ing1 {ing1_type})
  )
  :rule (then
    (achieve (agent-at ?t))
    ({action_name} {param_string})
  )
 )
 (:regression {action_name}-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type}))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {ing1_type})))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type})))))
  )
 )
"""

CW_REGRESSION_RULE_2 = r"""
 (:regression {rule_name}-1 [always]
  :parameters ((forall ?target-inventory - inventory) (forall ?target - object) (forall ?target-resource - object) (forall ?t - tile) (forall ?ing1 - object) (forall ?ing1-inventory - inventory) (forall ?ing2 - object) (forall ?ing2-inventory - inventory))
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type}))))
  :precondition (and
    (object-at ?target-resource ?t) (object-of-type ?target-resource {station_type}) (inventory-empty ?target-inventory) (object-of-type ?target Hypothetical)
    (inventory-holding ?ing1-inventory ?ing1) (object-of-type ?ing1 {ing1_type})
    (inventory-holding ?ing2-inventory ?ing2) (object-of-type ?ing2 {ing2_type})
  )
  :rule (then
    (achieve (agent-at ?t))
    ({action_name} {param_string})
  )
 )
 (:regression {rule_name}-2 [always]
  :goal (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type}))))
  :rule (then
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {ing1_type})))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {ing2_type})))))
    (achieve (exists (?i - inventory) (exists (?x - object) (and (inventory-holding ?i ?x) (object-of-type ?x {target_type})))))
  )
 )
 """


def _patch_cw_regression_rules(domain, verbose=False):
    import concepts.dsl.expression as E

    domain.incremental_define(CW_BASE_REGRESSION_RULES)
    for op_name, op in domain.operators.items():
        if op_name in ['move', 'pick-up', 'place-down']:
            continue

        inventory_varnames = [v.name for v in op.arguments if v.dtype.typename == 'inventory']
        object_varnames = [v.name for v in op.arguments if v.dtype.typename == 'object']
        tile_varnames = [v.name for v in op.arguments if v.dtype.typename == 'tile']

        if len(inventory_varnames) == 0:
            continue
        if len(object_varnames) == 0:
            continue
        if len(tile_varnames) == 0:
            continue

        object_to_inventory = {v: v for v in object_varnames}
        object_to_type = {v: None for v in object_varnames}
        is_location_object = {v: False for v in object_varnames}

        if verbose:
            print(op_name)
            print('  inventory_varnames:', inventory_varnames)
            print('  object_varnames:', object_varnames)
            print('  tile_varnames:', tile_varnames)

        for pre in op.preconditions:
            if isinstance(pre.bool_expr, E.FunctionApplicationExpression):
                expr = pre.bool_expr
                if (
                    expr.function.name == 'object-of-type' and
                    expr.arguments[0].__class__.__name__ == 'VariableExpression' and
                    expr.arguments[1].__class__.__name__ == 'ObjectConstantExpression'
                ):
                    object_to_type[expr.arguments[0].name] = expr.arguments[1].constant.name
                if (
                    expr.function.name == 'object-at' and
                    expr.arguments[0].__class__.__name__ == 'VariableExpression' and
                    expr.arguments[1].__class__.__name__ == 'VariableExpression'
                ):
                    is_location_object[expr.arguments[0].name] = True
                if (
                    expr.function.name == 'inventory-holding' and
                    expr.arguments[0].__class__.__name__ == 'VariableExpression' and
                    expr.arguments[1].__class__.__name__ == 'VariableExpression'
                ):
                    object_to_inventory[expr.arguments[1].name] = expr.arguments[0].name

        location_object = None
        for k, v in is_location_object.items():
            if v:
                location_object = k
        hypothetical_object = None
        for k, v in object_to_type.items():
            if v == 'Hypothetical':
                hypothetical_object = k
        ingredient_objects = list()
        for k, v in object_to_type.items():
            if v != 'Hypothetical' and k != location_object:
                ingredient_objects.append(k)

        target_type = None
        for effect in op.effects:
            if (
                effect.assign_expr.predicate.function.name == 'object-of-type' and
                effect.assign_expr.predicate.arguments[0].name == hypothetical_object and
                effect.assign_expr.predicate.arguments[1].__class__.__name__ == 'ObjectConstantExpression' and
                effect.assign_expr.value.__class__.__name__ == 'ConstantExpression' and
                effect.assign_expr.value.constant.item() == 1
            ):
                target_type = effect.assign_expr.predicate.arguments[1].name
            if (
                effect.assign_expr.predicate.function.name == 'inventory-holding' and
                effect.assign_expr.predicate.arguments[0].__class__.__name__ == 'VariableExpression' and
                effect.assign_expr.predicate.arguments[1].__class__.__name__ == 'VariableExpression' and
                effect.assign_expr.predicate.arguments[1].name == hypothetical_object and
                effect.assign_expr.value.__class__.__name__ == 'ConstantExpression' and
                effect.assign_expr.value.constant.item() == 1
            ):
                object_to_inventory[hypothetical_object] = effect.assign_expr.predicate.arguments[0].name

        if verbose:
            print('  location_object:', location_object)
            print('  hypothetical_object:', hypothetical_object)
            print('  ingredient_objects:', ingredient_objects)
            print('  target_type:', target_type)
            print('  object_to_inventory:', object_to_inventory)
            print('  object_to_type', object_to_type)

        if target_type is None:
            continue
        if location_object is None:
            continue
        station_type = object_to_type.get(location_object, None)
        if station_type is None:
            continue
        if None in object_to_inventory.values():
            continue
        if None in object_to_type.values():
            continue

        if verbose:
            print(' !!Start definition')

        try:
            if len(ingredient_objects) == 0:
                param_string = list()
                for p in op.arguments:
                    if p.name == location_object:
                        param_string.append('?target-resource')
                    elif p.dtype.typename == 'tile':
                        param_string.append('?t')
                    elif p.name == hypothetical_object:
                        param_string.append('?target')
                    elif p.name == object_to_inventory[hypothetical_object]:
                        param_string.append('?target-inventory')
                    else:
                        raise ValueError('Unknown parameter: {}'.format(p))
                param_string = ' '.join(param_string)
                rule = CW_REGRESSION_RULE_0.format(action_name=op_name, target_type=target_type, station_type=station_type, param_string=param_string)
            elif len(ingredient_objects) == 1:
                param_string = list()
                for p in op.arguments:
                    if p.name == location_object:
                        param_string.append('?target-resource')
                    elif p.dtype.typename == 'tile':
                        param_string.append('?t')
                    elif p.name == hypothetical_object:
                        param_string.append('?target')
                    elif p.name == object_to_inventory[hypothetical_object]:
                        param_string.append('?target-inventory')
                    elif p.name == ingredient_objects[0]:
                        param_string.append('?ing1')
                    elif p.name == object_to_inventory[ingredient_objects[0]]:
                        param_string.append('?ing1-inventory')
                    else:
                        raise ValueError('Unknown parameter: {}'.format(p))
                param_string = ' '.join(param_string)
                rule = CW_REGRESSION_RULE_1.format(action_name=op_name, target_type=target_type, station_type=station_type, ing1_type=object_to_type[ingredient_objects[0]], param_string=param_string)
            elif len(ingredient_objects) == 2:
                param_string = list()
                for p in op.arguments:
                    if p.name == location_object:
                        param_string.append('?target-resource')
                    elif p.dtype.typename == 'tile':
                        param_string.append('?t')
                    elif p.name == hypothetical_object:
                        param_string.append('?target')
                    elif p.name == object_to_inventory[hypothetical_object]:
                        param_string.append('?target-inventory')
                    elif p.name == ingredient_objects[0]:
                        param_string.append('?ing1')
                    elif p.name == object_to_inventory[ingredient_objects[0]]:
                        param_string.append('?ing1-inventory')
                    elif p.name == ingredient_objects[1]:
                        param_string.append('?ing2')
                    elif p.name == object_to_inventory[ingredient_objects[1]]:
                        param_string.append('?ing2-inventory')
                    else:
                        raise ValueError('Unknown parameter: {}'.format(p))
                param_string = ' '.join(param_string)
                rule = CW_REGRESSION_RULE_2.format(action_name=op_name, rule_name=op_name + '_order1', target_type=target_type, station_type=station_type, ing1_type=object_to_type[ingredient_objects[0]], ing2_type=object_to_type[ingredient_objects[1]], param_string=param_string)
                param_string = list()
                for p in op.arguments:
                    if p.name == location_object:
                        param_string.append('?target-resource')
                    elif p.dtype.typename == 'tile':
                        param_string.append('?t')
                    elif p.name == hypothetical_object:
                        param_string.append('?target')
                    elif p.name == object_to_inventory[hypothetical_object]:
                        param_string.append('?target-inventory')
                    elif p.name == ingredient_objects[1]:
                        param_string.append('?ing1')
                    elif p.name == object_to_inventory[ingredient_objects[1]]:
                        param_string.append('?ing1-inventory')
                    elif p.name == ingredient_objects[0]:
                        param_string.append('?ing2')
                    elif p.name == object_to_inventory[ingredient_objects[0]]:
                        param_string.append('?ing2-inventory')
                    else:
                        raise ValueError('Unknown parameter: {}'.format(p))
                param_string = ' '.join(param_string)
                rule += CW_REGRESSION_RULE_2.format(action_name=op_name, rule_name=op_name + '_order2', target_type=target_type, station_type=station_type, ing1_type=object_to_type[ingredient_objects[1]], ing2_type=object_to_type[ingredient_objects[0]], param_string=param_string)
            else:
                continue
        except ValueError as e:
            print('Error: {}'.format(e))
            continue

        # print(rule)
        domain.incremental_define(rule)



if __name__ == '__main__':
    main()

