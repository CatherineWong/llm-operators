import os
import copy

from llm_operators.pddl import Domain


def load_pddl_file_with_operators(domain_name, file_path, verbose=False):
    with open(os.path.join(file_path)) as f:
        raw_pddl = f.read()
    domain = Domain(pddl_domain=raw_pddl)
    domain.ground_truth_operators = {
        o: copy.deepcopy(domain.operators[o]) for o in domain.operators
    }
    if verbose:
        print('')
        print(f"Loaded PDDL file with operators")
        print('=' * 80)
        print(f'Domain: {domain_name}')
        print(f'Filename: {file_path}')
        print(f'Operators: {len(domain.operators)}')
        print("Ground truth operators: ")
        for o in list(domain.ground_truth_operators.keys()):
            print('  ', o)
    return domain
