"""
planning_domain.py | Classes for representing planning domains and planning domain datasets.
"""

from .dataset_core import Problem
from .dataset_core import load_pddl_supervision
from .dataset_core import PLANNING_PDDL_DOMAINS_REGISTRY, register_planning_pddl_domain, load_pddl_domain
from .dataset_core import PLANNING_PROBLEMS_REGISTRY, register_planning_domain_problems, get_problem_ids_with_ground_truth_operators, get_problem_ids_with_initial_plans_prefix, load_planning_problems_dataset

# Import all the planning domains.
from . import alfred  # noqa

# This can't be imported on Python 3.8
import sys
if sys.version_info.minor < 9:
    print("Warning: using Python < 3.9 to run ALFRED: cannot run craftingworld typing.")
else:    
    from . import crafting_world  # noqa