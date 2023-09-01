#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from llm_operators.datasets.crafting_world_gen.crafting_world_rules import MINING_RULES, CRAFTING_RULES

import networkx as nx
import matplotlib.pyplot as plt


def main():
    plt.figure(figsize=(12, 6))

    graph = nx.DiGraph()
    for rule in MINING_RULES:
        graph.add_edge(rule['location'], rule['create'])
        for x in rule['holding']:
            graph.add_edge(x, rule['create'])

    # two subfigures left and right
    plt.subplot(121)
    nx.draw_circular(graph, with_labels=True)
    plt.title("Mining Rules")
    plt.tight_layout()

    graph = nx.DiGraph()
    for rule in CRAFTING_RULES:
        for x in rule['recipe']:
            graph.add_edge(x, rule['create'])

    # visualize the graph
    plt.subplot(122)
    nx.draw_circular(graph, with_labels=True)
    plt.title("Crafting Rules")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

