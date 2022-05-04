#!/usr/bin/env python3
"""
@Project    ：SwarmIntelligenceOptimizationAlgorithm
@File       ：main.py 
@Author     ：Jaincen
@Time       ：2022/5/4 14:44
@Annotation : "main"
"""
from WOA import WOA
import matplotlib.pyplot as plt

if __name__ == "__main__":
    search_agents_no = 30
    max_iteration = 500
    # 对于函数F1
    lb = -100
    ub = 100
    dim = 40
    woa = WOA(search_agents_no, dim, lb, ub, max_iteration)
    cg_cover, best_score, best_pos = woa.woa()
    print("best score:", best_score)
    x = []
    for i in range(0, max_iteration):
        x.append(i)
    plt.plot(x, cg_cover)
    plt.show()
