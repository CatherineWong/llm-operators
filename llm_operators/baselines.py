import frozendict
from llm_operators import motion_planner


def _run_llm_propose_task_predicates_motion_planner(pddl_domain, problem_idx, problem_id, planning_problems, args, curr_iteration, output_directory, plan_pass_identifier, plan_attempt_idx, goal_idx, rng):
    problems = planning_problems["train"]

    any_motion_planner_success = False
    for idx, llm_task_plans in enumerate(problems[problem_id].proposed_pddl_task_predicates):
        print(f"Evaluating [{idx+1}/{len(problems[problem_id].proposed_pddl_task_predicates)}] LLM task plans.")
        # Sham task plans following the task planner format.
        task_plans = {
        problems[problem_id].ground_truth_pddl_problem.ground_truth_goal:  llm_task_plans
        }
        any_motion_planner_success, new_motion_plan_keys, used_motion_mock = motion_planner.attempt_motion_plan_for_problem(
                pddl_domain=pddl_domain,
                problem_idx=problem_idx,
                problem_id=problem_id,
                problems=planning_problems["train"],
                dataset_name=args.dataset_name,
                new_task_plans=task_plans,
                use_mock=args.debug_mock_motion_plans,
                command_args=args,
                curr_iteration=curr_iteration,
                output_directory=output_directory,
                plan_pass_identifier=plan_pass_identifier,
                plan_attempt_idx=plan_attempt_idx,
                resume=args.resume,
                resume_from_iteration=args.resume_from_iteration,
                resume_from_problem_idx=args.resume_from_problem_idx,
                debug_skip=args.debug_skip_motion_plans,
                verbose=args.verbose,
                llm_propose_task_predicates=args.llm_propose_task_predicates # Baseline -- we skip task proposal if so.
            )

    return any_motion_planner_success


def _run_llm_propose_code_policies_motion_planner(pddl_domain, problem_idx, problem_id, planning_problems, args, curr_iteration, output_directory, plan_pass_identifier, plan_attempt_idx, goal_idx, rng):
    

    #### TODO: REMOVE FOR DEBUGGING.
    problem_id = "train/pick_heat_then_place_in_recep-AppleSliced-None-SinkBasin-4/trial_T20190907_154132_135926"
    problems = planning_problems["train"]
    problems[problem_id].proposed_code_policies = [
        (frozendict.frozendict({
"action": "pick_up_object",
"argument_names" : ('env_state', 'env', 'object_id'),
"ground_arguments" : ("env_state", "env", "knife"),
"body" : """
    # Preconditions: None.

    # Low-level actions: pick up the object.
    try:
        action = PickupObject(args={'object_id':object_id})
        success = act(env=env, action=action)
        print("Successfully picked up an object!")
    except:
        print("Execution failure...")
        return False

    # Postconditions:
    final_env_state = perceive(env)
    return (final_env_state.holds(object_id)), final_env_state
"""
        }),
        frozendict.frozendict({
"action": "slice_object",
"argument_names" : ('env_state', 'env', 'object_id', 'tool_object_id'),
"ground_arguments" : ("env_state", "env", "apple", "knife"),
"body" : """
    # Preconditions: holding the tool object.
    if (not env_state.holds(tool_object_id)):
        print("Failure, not holding tool.")
        return False, env_state
    # Precondition: object is sliceable.
    if (not env_state.sliceable(object_id)):
        print("Failure, object is not sliceable.")
        return False, env_state

    # Low-level actions: slice the object if you're holding the tool.
    try:
        action = SliceObject(args={'object_id':object_id})
        success = act(env=env, action=action)
        print("Successfully sliced up an object!")
    except:
        print("Execution failure...")
        return False

    # Postconditions: the object is sliced.
    final_env_state = perceive(env)
    return (final_env_state.isSliced(object_id)), final_env_state
"""
        }),
        frozendict.frozendict({
                "action": "put_object_in_receptacle",
                "argument_names" : ('env_state', 'env', 'object_id', 'receptacle_object_id'),
                "ground_arguments" : ("env_state", "env", "knife", "diningtable"),
                "body" : """
    # Preconditions: we should be holding the object.
    if (not env_state.holds(object_id)):
        print("Failure, not holding object.")
        return False, env_state

    # Low-level actions: put the object in the receptacle.
    try:
        print("Trying to put in receptacle.")
        action = PutObject(args={'object_id':object_id, 'receptacle_object_id':receptacle_object_id})
        success = act(env=env, action=action)
        print(f"Result of PutObject is: {success}")
    except:
        print("Failure, could not place in receptacle.")
        return False, env_state

    # Postconditions:
    final_env_state = perceive(env)
    if (not final_env_state.holds(object_id)) and (final_env_state.inReceptacle(object_id, receptacle_object_id)):
        return True, final_env_state
                """
        }),
        frozendict.frozendict({
"action": "pick_up_object",
"argument_names" : ('env_state', 'env', 'object_id'),
"ground_arguments" : ("env_state", "env", "apple"),
"body" : """
    # Preconditions: None.

    # Low-level actions: pick up the object.
    try:
        action = PickupObject(args={'object_id':object_id})
        success = act(env=env, action=action)
        print("Successfully picked up an object!")
    except:
        print("Execution failure...")
        return False

    # Postconditions:
    final_env_state = perceive(env)
    return (final_env_state.holds(object_id)), final_env_state
"""
        }),
        frozendict.frozendict({
"action": "cool_object",
"argument_names" : ('env_state', 'env', 'object_id', 'receptacle_object_id'),
"ground_arguments" : ("env_state", "env", "apple", "fridge"),
"body" : """
    # Preconditions: we should be holding the object.
    if (not env_state.holds(object_id)):
        print("Failure, not holding object.")
        return False, env_state

    # Low-level actions: open the receptacle
    if (not env_state.opened(receptacle_object_id)):
        try:
            action = OpenObject(args={'object_id':receptacle_object_id})
            success = act(env=env, action=action)
            print("Successfully opened up an object!")
        except:
            print("Execution failure...")
            return False

    # Low-level actions: put the object in the receptacle.
    try:
        print("Trying to put in receptacle.")
        action = PutObject(args={'object_id':object_id, 'receptacle_object_id':receptacle_object_id})
        success = act(env=env, action=action)
        print(f"Result of PutObject is: {success}")
    except:
        print("Failure, could not place in receptacle.")
        return False, env_state

    # Low-level actions: close the receptacle
    try:
        action = CloseObject(args={'object_id':receptacle_object_id})
        success = act(env=env, action=action)
        print(f"Result of CloseObject is: {success}")
    except:
        print("Execution failure...")
        return False

    # Low-level actions: open the microwave.
    if (not env_state.opened(receptacle_object_id)):
        try:
            action = OpenObject(args={'object_id':receptacle_object_id})
            success = act(env=env, action=action)
            print(f"Result of OpenObject is: {success}")
        except:
            print("Execution failure...")
            return False


    # Postconditions:
    final_env_state = perceive(env)
    return (final_env_state.isCool(object_id)), final_env_state
"""
        }),
        )
    ]

    any_motion_planner_success = False
    for idx, llm_code_policies in enumerate(problems[problem_id].proposed_code_policies):
        print(f"Evaluating [{idx+1}/{len(problems[problem_id].proposed_code_policies)}] LLM proposed code policies.")
        # Sham task plans following the task planner format.
        code_policies = {
        problems[problem_id].ground_truth_pddl_problem.ground_truth_goal:  llm_code_policies
        }
        any_motion_planner_success, new_motion_plan_keys, used_motion_mock = motion_planner.attempt_motion_plan_for_problem(
                pddl_domain=pddl_domain,
                problem_idx=problem_idx,
                problem_id=problem_id,
                problems=planning_problems["train"],
                dataset_name=args.dataset_name,
                new_task_plans=code_policies,
                use_mock=args.debug_mock_motion_plans,
                command_args=args,
                curr_iteration=curr_iteration,
                output_directory=output_directory,
                plan_pass_identifier=plan_pass_identifier,
                plan_attempt_idx=plan_attempt_idx,
                resume=args.resume,
                resume_from_iteration=args.resume_from_iteration,
                resume_from_problem_idx=args.resume_from_problem_idx,
                debug_skip=args.debug_skip_motion_plans,
                verbose=args.verbose,
                llm_propose_code_policies=args.llm_propose_code_policies # Baseline -- we skip task proposal if so.
            )

    return any_motion_planner_success
