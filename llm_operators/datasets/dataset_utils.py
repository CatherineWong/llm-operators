import os
import copy

from pddl import Domain


def load_pddl_file_with_operators(domain_name, file_path, verbose=False):
    with open(os.path.join(file_path)) as f:
        raw_pddl = f.read()
    domain = Domain(pddl_domain=raw_pddl)
    domain.ground_truth_operators = {
        o: copy.deepcopy(domain.operators[o]) for o in domain.operators
    }
    if verbose:
        print(f"\nload_pddl_file_with_operators: loaded {domain_name} from {file_path}")
        print("\nGround truth operators: ")
        for o in list(domain.ground_truth_operators.keys()):
            print(o)
    return domain
