#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import jacinle
import os.path as osp

from llm_operators.datasets.crafting_world_gen.cw_20230204_minining_only import gen_v20230204_instance_record, problem_from_raw_record

parser = jacinle.JacArgumentParser()
parser.add_argument('--data-dir', type=str, default='data/dataset/crafting_world_v20230204_mining_only')
parser.add_argument('--train-size', type=int, default=100)
parser.add_argument('--valid-size', type=int, default=10)

parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--embed', action='store_true', help='enter IPython embed after running the script.')
args = parser.parse_args()


def main():
    jacinle.seed(args.seed)
    jacinle.io.set_fs_verbose()

    output = dict(train=[], valid=[])
    output_problems = dict(train=[], valid=[])
    for i in range(args.train_size):
        output['train'].append(gen_v20230204_instance_record(f'train_{i}', 'train'))
        output_problems['train'].append(problem_from_raw_record(output['train'][-1]))
    for i in range(args.valid_size):
        output['valid'].append(gen_v20230204_instance_record(f'valid_{i}', 'valid'))
        output_problems['valid'].append(problem_from_raw_record(output['valid'][-1]))

    jacinle.mkdir(args.data_dir)
    jacinle.dump(osp.join(args.data_dir, 'dataset.json'), output)

    if args.embed:
        import IPython
        IPython.embed()


if __name__ == '__main__':
    main()
