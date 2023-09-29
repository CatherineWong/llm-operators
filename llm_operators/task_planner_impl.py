from tempfile import NamedTemporaryFile

from pddlgym_planners.fd import FD
from pddlgym_planners.planner import PlanningFailure, PlanningTimeout

TASK_PLANNER_FD_DEFAULT_TIMEOUT = 3
TASK_PLANNER_PDSKETCH_ONTHEFLY_DEFAULT_TIMEOUT = 10
TASK_PLANNER_PDSKETCH_REGRESSION_DEFAULT_TIMEOUT = 10


def fd_plan_from_strings(domain_str, problem_str, timeout=None, verbose=False):
    if timeout is None:
        timeout = TASK_PLANNER_FD_DEFAULT_TIMEOUT

    with NamedTemporaryFile(mode="w") as domain_file, NamedTemporaryFile(mode="w") as problem_file:
        domain_file.write(domain_str)
        problem_file.write(problem_str)
        domain_file.flush()
        problem_file.flush()
        success, out = fd_plan_from_file(domain_file.name, problem_file.name, timeout=timeout)
        return (success, out)


def fd_plan_from_file(domain_fname, problem_fname, timeout=None):
    if timeout is None:
        timeout = TASK_PLANNER_FD_DEFAULT_TIMEOUT

    # TBD: don't use PDDL gym planner, use original FD.
    fd_planner = FD(alias_flag='--alias "lama-first"')
    try:
        plan = fd_planner.plan_from_pddl(domain_fname, problem_fname, timeout=timeout)
        plan_string = "\n".join(["(" + a + ")" for a in plan])
    except PlanningFailure as pf:
        return False, pf
    except PlanningTimeout as pt:
        print("Time out")
        return False, pt
    return True, plan_string


def pdsketch_onthefly_plan_from_strings(domain_str, problem_str, timeout=None, heuristic=None):
    if timeout is None:
        timeout = TASK_PLANNER_PDSKETCH_ONTHEFLY_DEFAULT_TIMEOUT

    import concepts.pdsketch as pds

    domain = pds.load_domain_string(domain_str)
    problem = pds.load_problem_string(problem_str, domain, return_tensor_state=False)

    from concepts.pdsketch.strips.strips_grounding_onthefly import (
        OnTheFlyGStripsProblem,
    )

    gproblem = OnTheFlyGStripsProblem.from_domain_and_problem(domain, problem)
    # import ipdb; ipdb.set_trace()

    if heuristic is None:
        from concepts.pdsketch.strips.strips_grounding_onthefly import ogstrips_search

        plan = ogstrips_search(gproblem, timeout=timeout, initial_actions=[])
    elif heuristic == "hmax":
        from concepts.pdsketch.strips.strips_grounding_onthefly import ogstrips_search_with_heuristics

        # plan = ['move-right(t1, t2)', 'move-right(t2, t3)', 'move-right(t3, t4)', 'move-right(t4, t5)', 'move-right(t5, t6)', 'move-right(t6, t7)', 'move-right(t7, t8)', 'move-right(t8, t9)', 'pick-up(t9, o5, i2)', 'move-right(t9, t10)', 'harvest-sugar-cane(i3, t10, t0, o5, o10, i2, o17)']
        # canonized_plan = _pdsketch_get_canonized_plan(gproblem, plan)
        # plan = ogstrips_search_with_heuristics(gproblem, initial_actions=canonized_plan, timeout=timeout, hfunc_name='hmax', verbose=True, hfunc_verbose=True)
        plan = ogstrips_search_with_heuristics(gproblem, timeout=timeout, hfunc_name="hmax", g_weight=0.5)
    elif heuristic == "hff":
        from concepts.pdsketch.strips.strips_grounding_onthefly import ogstrips_search_with_heuristics

        plan = ogstrips_search_with_heuristics(gproblem, timeout=timeout, hfunc_name="hff", g_weight=0)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")

    if plan is None:
        return False, None
    return (
        True,
        "\n".join([op.to_applier_pddl_str(arguments) for op, arguments in plan]),
    )


def pdsketch_onthefly_verify_plan_from_strings(domain_str, problem_str, plan):
    import concepts.pdsketch as pds

    domain = pds.load_domain_string(domain_str)
    problem = pds.load_problem_string(problem_str, domain, return_tensor_state=False)

    from concepts.pdsketch.strips.strips_grounding_onthefly import (
        OnTheFlyGStripsProblem,
    )

    gproblem = OnTheFlyGStripsProblem.from_domain_and_problem(domain, problem)

    from concepts.pdsketch.strips.strips_grounding_onthefly import ogstrips_verify

    ogstrips_verify(gproblem, [action.lower() for action in plan], from_fast_downward=True)


def _pdsketch_get_canonized_plan(gproblem, plan_strings):
    canonized_plan = list()
    for action in plan_strings:
        action_name = action.split("(")[0]
        action_args = action.split("(")[1].split(")")[0].split(", ")
        operator = gproblem.operators[action_name]
        canonized_plan.append((operator, {arg.name: value for arg, value in zip(operator.arguments, action_args)}))

    return canonized_plan


def pdsketch_regression_plan_from_strings(domain_str, problem_str, timeout=None, heuristic=None):
    if timeout is None:
        timeout = TASK_PLANNER_PDSKETCH_ONTHEFLY_DEFAULT_TIMEOUT

    import concepts.pdsketch as pds

    domain = pds.load_domain_string(domain_str)
    _patch_cw_regression_rules(domain)
    executor = pds.PDSketchExecutor(domain)

    initial_state, goal = pds.load_problem_string(problem_str, executor.domain, executor=executor)
    from concepts.pdsketch.planners.optimistic_search_bilevel_legacy import enumerate_possible_symbolic_plans_regression_c_v2
    rv, stat = enumerate_possible_symbolic_plans_regression_c_v2(executor, initial_state, goal, enable_reordering=False, verbose=False)

    if len(rv) == 0:
        return False, None
    return True, "\n".join([op.pddl_str() for op in rv[0][0]])


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


def _patch_cw_regression_rules(domain):
    import concepts.dsl.expression as E

    domain.incremental_define(CW_BASE_REGRESSION_RULES)
    for op_name, op in domain.operators.items():
        if op_name in ['move', 'pick-up']:
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

        # print(op_name)
        # print('  inventory_varnames:', inventory_varnames)
        # print('  object_varnames:', object_varnames)
        # print('  tile_varnames:', tile_varnames)

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

        # print('  location_object:', location_object)
        # print('  hypothetical_object:', hypothetical_object)
        # print('  ingredient_objects:', ingredient_objects)
        # print('  target_type:', target_type)
        # print('  object_to_inventory:', object_to_inventory)
        # print('  object_to_type', object_to_type)

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
            # print('Error: {}'.format(e))
            continue

        domain.incremental_define(rule)

