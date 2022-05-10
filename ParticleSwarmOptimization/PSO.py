#!/usr/bin/env python3
"""
@Project    ：SwarmIntelligenceOptimizationAlgorithm 
@File       ：PSO.py 
@Author     ：Jaincen
@Time       ：2022/5/10 19:05
@Annotation : "Particle Swarm Optimization"
"""
import numpy as np
import sys
from Common import test_function


class PSO:
    def __init__(self, search_agents_no, dim, lb, ub, max_iter):
        self.search_agents_no = search_agents_no
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.max_iter = max_iter

    def initiallization(self):
        # 待修改
        return np.random.rand(self.search_agents_no, self.dim) * (self.ub - self.lb) + self.lb

    def pso(self):

        Vmax = 6
        noP = self.search_agents_no
        wMax = 0.9
        wMin = 0.2
        c1 = 2
        c2 = 2
        ub = self.ub
        lb = self.lb

        max_iter = self.max_iter
        vel = np.zeros((noP, self.dim))
        p_best_score = np.zeros(noP)
        p_best_score.fill(sys.maxsize)
        p_best_position = np.zeros(self.dim)
        p_best = np.zeros((noP, self.dim))

        g_best = np.zeros(self.dim)
        g_best_score = sys.maxsize

        convergence_curve = np.zeros(max_iter)

        positions = self.initiallization()

        for t in range(0, max_iter):

            for i in range(0, positions.shape[0]):
                flag4ub = positions[i] > ub
                flag4lb = positions[i] < lb
                positions[i] = positions[i] * (~(flag4lb + flag4ub)) + ub * flag4ub + lb * flag4lb

                fitness = test_function.f1(positions[i])
                if fitness < p_best_score[i]:
                    p_best_score[i] = fitness
                    p_best[i] = positions[i].copy()
                if fitness < g_best_score:
                    g_best = positions[i].copy()
                    g_best_score = fitness
            w = wMax - t * ((wMax - wMin) / max_iter)

            for i in range(0, positions.shape[0]):
                for j in range(0, positions.shape[1]):
                    vel[i][j] = w * vel[i][j] + c1 * np.random.rand() * (p_best[i][j] - positions[i][j]) + \
                                c2 * np.random.rand() * (g_best[j] - positions[i][j])
                    if vel[i][j] > Vmax:
                        vel[i][j] = Vmax
                    if vel[i][j] < -Vmax:
                        vel[i][j] = -Vmax
                    positions[i][j] = positions[i][j] + vel[i][j]

            convergence_curve[t] = g_best_score
        return convergence_curve, g_best_score, g_best