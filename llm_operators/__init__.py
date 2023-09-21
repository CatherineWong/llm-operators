#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path as osp
import sys

# Import Jacinle.
JACINLE_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "../../jacinle")
print("Adding jacinle path: {}".format(JACINLE_PATH))
sys.path.insert(0, JACINLE_PATH)

# Import Concepts.
CONCEPTS_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "../../concepts")
print("Adding concepts path: {}".format(CONCEPTS_PATH))
sys.path.insert(0, CONCEPTS_PATH)