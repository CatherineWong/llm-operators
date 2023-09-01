#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import jacinle
import argparse
from llm_operators.task_planner import pdsketch_onthefly_plan_from_strings, pdsketch_onthefly_verify_plan_from_strings
from llm_operators.task_planner import fd_plan_from_strings

parser = argparse.ArgumentParser(description='Test the planner')
parser.add_argument('task', type=str, choices=['bf', 'verify', 'hmax', 'hff'], help='Task name')
parser.add_argument('domain', type=str, help='domain file')
parser.add_argument('problem', type=str, help='problem file')
parser.add_argument('--timeout', type=int, default=10, help='timeout in seconds')
args = parser.parse_args()


def main_verify():
    with open(args.domain, 'r') as f:
        domain_string = f.read()
    with open(args.problem, 'r') as f:
        problem_string = f.read()

    print('Planning with Fast Downward')
    _, plan = fd_plan_from_strings(domain_string, problem_string, timeout=args.timeout)
    print(plan)

    print('-' * 80)

    print('Verify with PDSketch')
    pdsketch_onthefly_verify_plan_from_strings(domain_string, problem_string, plan)

    print('-' * 80)

    print('Planning with PDSketch')
    _, plan = pdsketch_onthefly_plan_from_strings(domain_string, problem_string, timeout=args.timeout, heuristic='hmax')
    print(plan)

    print('-' * 80)

    print('Verify (again) with PDSketch')
    pdsketch_onthefly_verify_plan_from_strings(domain_string, problem_string, plan)


def main_heuristic(heuristic):
    with open(args.domain, 'r') as f:
        domain_string = f.read()
    with open(args.problem, 'r') as f:
        problem_string = f.read()

    print('Planning with PDSketch')
    _, plan = pdsketch_onthefly_plan_from_strings(domain_string, problem_string, timeout=args.timeout, heuristic=heuristic)
    print(plan)


def main_bf():
    with open(args.domain, 'r') as f:
        domain_string = f.read()
    with open(args.problem, 'r') as f:
        problem_string = f.read()

    print('Planning with PDSketch')
    _, plan = pdsketch_onthefly_plan_from_strings(domain_string, problem_string, timeout=args.timeout)
    print(plan)


if __name__ == '__main__':
    with jacinle.profile():
        if args.task == 'verify':
            main_verify()
        elif args.task == 'hmax':
            main_heuristic('hmax')
        elif args.task == 'hff':
            main_heuristic('hff')
        elif args.task == 'bf':
            main_bf()
        else:
            raise ValueError('Unknown task: {}'.format(args.task))

