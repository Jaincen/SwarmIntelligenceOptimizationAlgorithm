#!/usr/bin/env python3
"""
@Project    ：SwarmIntelligenceOptimizationAlgorithm 
@File       ：main.py 
@Author     ：Jaincen
@Time       ：2022/5/10 19:32
@Annotation : "main"
"""
from WhaleOptimizationAlgorithm.WOA import WOA
from ParticleSwarmOptimization.PSO import PSO
import matplotlib.pyplot as plt

if __name__ == "__main__":
    search_agents_no = 30
    max_iteration = 500
    # 对于函数F1
    lb = -100
    ub = 100
    dim = 30
    woa = WOA(search_agents_no, dim, lb, ub, max_iteration)
    pso = PSO(search_agents_no, dim, lb, ub, max_iteration)
    woa_cg_cover, woa_best_score, woa_best_pos = woa.woa()
    pso_cg_cover, pso_best_score, pso_best_pos = pso.pso()
    x = []
    print("WOA best: ", woa_best_score)
    print("PSO best: ", pso_best_score)
    for i in range(0, max_iteration):
        x.append(i)
    plt.plot(x, woa_cg_cover, label = "WOA")
    plt.plot(x, pso_cg_cover, label="PSO")
    plt.yscale('log')
    plt.legend()
    plt.show()