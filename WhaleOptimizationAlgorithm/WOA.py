#!/usr/bin/env python3
"""
@Project    ：SwarmIntelligenceOptimizationAlgorithm
@File       ：WOA.py 
@Author     ：Jaincen
@Time       ：2022/5/4 14:44
@Annotation :"Whale Optimization Algorithm For Python"
"""
import numpy as np
import sys
import math
import test_function


class WOA:
    def __init__(self, search_agents_no, dim, lb, ub, max_iter):
        self.search_agents_no = search_agents_no
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.max_iter = max_iter

    def initiallization(self):
        # 待修改
        positions = np.random.rand(self.search_agents_no, self.dim) * (self.ub - self.lb) + self.lb
        return positions

    def woa(self):
        leader_pos = np.zeros(self.dim)
        leader_score = sys.maxsize
        positions = self.initiallization()
        convergence_curve = np.zeros(self.max_iter)
        for t in range(0, self.max_iter):
            for i in range(0, positions.shape[0]):
                flag4ub = positions[i] > self.ub
                flag4lb = positions[i] < self.lb
                positions[i] = positions[i] * (~(flag4lb+flag4ub))+self.ub*flag4ub+self.lb*flag4lb

                fitness = test_function.f1(positions[i])
                if fitness < leader_score:
                    leader_score = fitness
                    leader_pos = positions[i]

            a = 2 - t * (2/self.max_iter)
            a2 = -1 + t*(-1/self.max_iter)

            for i in range(0, positions.shape[0]):
                r1 = np.random.rand()
                r2 = np.random.rand()

                A = 2*a*r1-a
                C = 2*r2
                b = 1
                l = (a2 - 1) * np.random.rand()+1
                p = np.random.rand()

                for j in range(0, positions.shape[1]):
                    if p < 0.5:
                        if abs(A) >= 1:
                            rand_leader_index = math.floor(self.search_agents_no*np.random.rand())
                            x_rand = positions[rand_leader_index]
                            d_x_rand = abs(C*x_rand[j] - positions[i][j])
                            positions[i][j] = x_rand[j]-A*d_x_rand
                        elif abs(A) < 1:
                            d_leader = abs(C*leader_pos[j]-positions[i][j])
                            positions[i][j] = leader_pos[j] - A*d_leader

                    elif p >= 0.5:
                        distance2leader = abs(leader_pos[j] - positions[i][j])
                        positions[i][j] = distance2leader * np.exp(b*l) * np.cos(l*2*math.pi) + leader_pos[j]

            convergence_curve[t] = leader_score
        return convergence_curve, leader_score, leader_pos