import numpy as np
import copy 
from tqdm import tqdm
from datahandler import *
from ea_helpers import *
from logger import *

class NSGA2:

    def __init__(self, data_handler: Data_Handler):
        self.n = data_handler.n
        self.k = data_handler.k
        self.d = data_handler.d
        self.T = data_handler.T
        self.S = data_handler.S
        self.data_handler = data_handler
        pass
    
    def get_random_solution_population(self,N):
        P = []
        for i in range(N):
            A = random_matrix((self.d,self.k))
            A = gram_schmidt_algorithm(A)
            beta = self.data_handler.linear_l2_regression(A)
            P.append(Solution(beta,A))
        return P

    def get_binary_tournament_selection(self,sample_size,N,p_t):
        i = 0
        p_tresh = p_t
        sample = set()
        while len(sample) < sample_size and i < N:
            if np.random.random() < p_tresh:
                sample.add(i)
            p_tresh = p_tresh*(1-p_t)
            i+=1

        while len(sample) < sample_size:
            sample.add(np.random.randint(0,N))

        return sample
    
    def get_offspring_population(self, N, P, p_c,p_m, p_t,mutation_step_size,use_reg_for_crossover):
        C = []
        for i in range(N/2):
            sids = self.get_binary_tournament_selection(2,N,p_t)
            S1, S2 = P[sids[0]], P[sids[1]]

            if np.random.random() < p_c:
                S1,S2 = self.crossover(S1,S2,use_reg_for_crossover)
            if np.random.random() < p_m:
                S1 = self.mutate(S1,mutation_step_size)
                S2 = self.mutate(S2,mutation_step_size)
        return C

    def run(self, N, G, p_c, p_m,p_t, mutation_step_size,crossover_step_size,use_reg_for_crossover:bool):
        """
        p_t ~ 1/n add a factor of n to algorithm i.e. 1/p_t so keep p_t constant
        """
        assert(N%2==0)
        P = self.get_random_solution_population(N)
        for p in P:
            _R = self.data_handler.compute_reward(p.beta,p.A)
            _L = self.data_handler.compute_loss(p.beta,p.A)
            p.first_set_RL_NSGA(_R,_L)
        P = non_dominated_sorting(P,True)
        for gen in tqdm(range(G), desc="Generation of NSGA2"):
            C = self.get_offspring_population(N, P, p_c, p_m,p_t,mutation_step_size,crossover_step_size, use_reg_for_crossover)
            for c in C:
                _R = self.data_handler.compute_reward(c.beta,c.A)
                _L = self.data_handler.compute_loss(c.beta,c.A)
                c.first_set_RL_NSGA(_R,_L)
            F_list = non_dominated_sorting(P+C)
            P = []
            n_gen = len(F_list)
            i = 0
            while i < n_gen and len(P)+len(F_list[i]) < N:
                P += F_list[i]
                i += 1
            if len(P) < N and i < n_gen:
                F_list[i] = compute_crowding_distance(F_list[i],True)
                sorted(F_list[i], key = lambda p:p.crowding_distance)
                P += F_list[i][:N-len(P)]
        return P

    def crossover(self,S1:Solution,S2:Solution,crossover_step_size,use_reg_for_crossover):
        C1, C2 = copy.deepcopy(S1), copy.deepcopy(S2)
        deltaA = crossover_step_size*(C2.A -C1.A)
        T_1 =  0.5*(np.matmul(deltaA.T,C1.A) - np.matmul(C1.A.T,deltaA))
        T_2 =  0.5*(-np.matmul(deltaA.T,C2.A) + np.matmul(C2.A.T,deltaA))
        Q1 = cayley_transformation(T_1)
        Q2 = cayley_transformation(T_2)
        C1.A = C1.A*Q1
        C2.A = C2.A*Q2

        deltabeta = crossover_step_size*(C2.beta -C1.beta)
        if use_reg_for_crossover:
            C1.beta = self.data_handler.linear_l2_regression(C1.A)
            C2.beta = self.data_handler.linear_l2_regression(C2.A)
        else:
            C1.beta = C1.beta + deltabeta
            C2.beta = C2.beta - deltabeta
        return C1,C2

    def mutate(self,S:Solution,mutation_step_size):
        Sm = copy.deepcopy(S)
        deltabeta = random_matrix((self.k,1))
        deltaA = random_matrix((self.d,self.k))
        T_ =  0.5*mutation_step_size*(np.matmul(deltaA.T,Sm.A) - np.matmul(Sm.A.T,deltaA))
        Q =  cayley_transformation(T_)
        Sm.A = Sm.A*Q
        Sm.beta = Sm.beta+mutation_step_size*deltabeta
        return Sm