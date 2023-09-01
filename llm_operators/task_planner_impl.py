from tempfile import NamedTemporaryFile

from pddlgym_planners.fd import FD
from pddlgym_planners.planner import PlanningFailure, PlanningTimeout

TASK_PLANNER_FD_DEFAULT_TIMEOUT = 10
TASK_PLANNER_PDSKETCH_ONTHEFLY_DEFAULT_TIMEOUT = 10


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
