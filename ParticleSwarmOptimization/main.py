#!/usr/bin/env python3
"""
@Project    ：SwarmIntelligenceOptimizationAlgorithm 
@File       ：main.py 
@Author     ：Jaincen
@Time       ：2022/5/10 19:07
@Annotation : "main for Particle Swarm Optimization"
"""
from PSO import PSO
import matplotlib.pyplot as plt


if __name__ == "__main__":
    SearchAgents_no = 30
    max_iteration = 500
    lb = -100
    ub = 100
    dim = 30
    pso = PSO(SearchAgents_no, dim, lb, ub, max_iteration)
    cg_cover, best_score, best_pos = pso.pso()
    print("best score:", best_score)
    x = []
    for i in range(0, max_iteration):
        x.append(i)
    plt.plot(x, cg_cover)
    plt.yscale('log')
    plt.show()
