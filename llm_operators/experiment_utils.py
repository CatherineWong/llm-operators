import os
import sys
import json
import pathlib
import contextlib
import time
import datetime
from llm_operators import pddl

# TODO(Jiayuan Mao @ 2023/02/04): use a principled way to control the random seed.
RANDOM_SEED = 0


class Unbuffered:
    def __init__(self, stream, txt_file):
        self.stream = stream
        self.txt_file = txt_file

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.txt_file.write(data)
        self.txt_file.flush()


def _custom_exception_hook(type, value, tb):
    with temporarily_release_print():
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type, value, tb)
        else:
            import traceback, ipdb
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type, value, tb)
            # ...then start the debugger in post-mortem mode.
            ipdb.post_mortem(tb)


def hook_exception_ipdb():
    """Add a hook to ipdb when an exception is raised."""
    if not hasattr(_custom_exception_hook, 'origin_hook'):
        _custom_exception_hook.origin_hook = sys.excepthook
        sys.excepthook = _custom_exception_hook


def run_ipdb():
    import ipdb
    with temporarily_release_print():
        ipdb.set_trace()


def should_use_checkpoint(curr_iteration, curr_problem_idx, resume_from_iteration, resume_from_problem_idx):
    # Should we attempt to load from a checkpoint?
    if curr_iteration <= resume_from_iteration:
        if curr_problem_idx is None or curr_problem_idx <= resume_from_problem_idx:
            return True
    return False


def get_output_directory(curr_iteration, command_args, experiment_name_to_load):
    output_directory = command_args.output_directory

    if curr_iteration is None:
        full_output_directory = os.path.join(output_directory, experiment_name_to_load)
    else:
        full_output_directory = os.path.join(output_directory, experiment_name_to_load, str(curr_iteration))
    pathlib.Path(full_output_directory).mkdir(parents=True, exist_ok=True)
    return full_output_directory


def redirect_print(command_args, experiment_name_to_load):
    output_directory = get_output_directory(None, command_args, experiment_name_to_load)
    filename = os.path.join(output_directory, time.strftime("%Y%m%d-%H%M%S") + ".log")
    print('Redirecting print to', filename)
    sys.stdout = Unbuffered(sys.stdout, open(filename, "w"))


@contextlib.contextmanager
def temporarily_release_print():
    backup = sys.stdout
    sys.stdout = sys.__stdout__
    yield
    sys.stdout = backup


def output_iteration_summary(
    curr_iteration, pddl_domain, problems, command_args, output_directory, finished_epoch, problem_idx, total_problems
):
    # Print the total experiment progress.
    print('Experiment Summary')
    print("=" * 80)
    print(f"Iteration: {curr_iteration}")
    print(f"Evaluated successful motion plans: {len([p for p in list(problems.values())[:problem_idx + 1] if len(p.solved_motion_plan_results) > 0])} / {problem_idx + 1} problems so far.")
    print(f'Overall successful motion plans: {len([p for p in list(problems.values()) if len(p.solved_motion_plan_results) > 0])} / {len(problems)} problems.')
    if finished_epoch:
        for p in problems.values():
            if len(p.solved_motion_plan_results) > 0:
                plan_string = list(p.solved_motion_plan_results.values())[0].pddl_plan.plan_string.replace('\n', ' ')
                print(f"  SOLVED - {p.language}")
                print(f"  {plan_string}")
            else:
                print(f"  UNSOLVED - {p.language}")
    print(f"Total problems: {len(problems)}")
    print(f"Current operators: {len(problems)}")
    print('')


def output_experiment_parameters(command_args):
    command_args_string = ""
    for command, args in vars(command_args).items():
        if type(args) == list:
            args_str = " ".join(args)
        elif type(args) == bool:
            args_str = ""  # action=store_true
        else:
            args_str = args
        command_args_string += f"--{command} {args_str} "

    print('Experiment Parameters')
    print('=' * 80)
    print(f"Timestamp: {datetime.datetime.now()}")
    print(f"Command to replicate: python main.py {command_args_string}")
    print('')

    directory = get_output_directory(None, command_args, command_args.experiment_name)
    filename = os.path.join(directory, "command_args.json")
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump(vars(command_args), f, indent=4)
    filename = os.path.join(directory, "command_args.txt")
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write(' '.join(sys.argv))
            f.write(command_args_string)

from ast import literal_eval
def resume_from_last_scored_operators(experiment_name, pddl_domain, previous_output_directory,  operator_acceptance_threshold,
    operator_pseudocounts):
    experiment_tag = "" if len(experiment_name) < 1 else f"{experiment_name}_"
    output_filepath = f"{experiment_tag}scored_operators.json"
    with open(os.path.join(previous_output_directory, output_filepath), "r") as f:
            operators_to_scores = json.load(f)
    for str_o_name_body in operators_to_scores:
        scores = operators_to_scores[str_o_name_body]
        o_name, o_body = literal_eval(str_o_name_body)
        pddl_domain.operators_to_scores[(o_name, o_body)] = scores
    
    # Set operators with final scores.
    for o_name, o_body in pddl_domain.operators_to_scores:
        (o_success, o_attempts) = pddl_domain.operators_to_scores[(o_name, o_body)]
        p_success = float(o_success / o_attempts)
        if (
            p_success > operator_acceptance_threshold and o_attempts > operator_pseudocounts
        ):  # And we've used it at least once beyond the pseduocounts.
            pddl_domain.add_operator(operator_name=o_name, operator_pddl=o_body)
    print(f"Final operators after iteration: {pddl_domain.operators.keys()}")