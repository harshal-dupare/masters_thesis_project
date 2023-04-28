import numpy as np
from tqdm import tqdm
import copy
from datahandler import *
from ea_helpers import *
from logger import *

class MOPSO:
    def __init__(self, data_handler: Data_Handler):
        self.n = data_handler.n
        self.k = data_handler.k
        self.d = data_handler.d
        self.T = data_handler.T
        self.S = data_handler.S
        self.data_handler = data_handler
        pass

    def initialize_population_and_velocity(self,N):
        P = []
        for i in range(N):
            A = random_matrix((self.d,self.k))
            A = gram_schmidt_algorithm(A)
            beta = self.data_handler.linear_l2_regression(A)
            vel_beta = self.data_handler.compute_reward_gradient_beta(beta, A)
            vel_A = self.data_handler.compute_reward_gradient_A(beta, A)
            P+=[Solution(beta,A,vel_beta,vel_A)]
        return P

    def update_velocity(self, p:Solution, best_global:Solution,w,c_1,c_2):
        r1, r2 = np.random.random(), np.random.random()
        p.vel_beta = w*best_global.vel_beta+c_1*r1*(p.best_beta-p.beta)+c_2*r2*(best_global.best_beta-p.beta)
        p.vel_A = w*best_global.vel_A+c_1*r1*(p.best_A-p.A)+c_2*r2*(best_global.best_A-p.A)
        pass

    def update_position(self, p:Solution, eps):
        T_ =  0.5*eps*(np.matmul(p.vel_A.T, p.A) - np.matmul(p.A,p.vel_A.T))
        Q = cayley_transformation(T_)
        p.A = p.A*Q
        p.beta = p.beta + eps*p.vel_beta
        pass

    def mutate(self, p:Solution, mutation_step_size, vel_mutation_setp_size):
        Sm = copy.deepcopy(S)
        deltabeta = random_matrix((self.k,1))
        deltaA = random_matrix((self.d,self.k))
        T_ = 0.5*mutation_step_size*(np.matmul(deltaA.T, Sm.A) - np.matmul(Sm.A.T,deltaA))
        Q =  cayley_transformation(T_)
        Sm.A = Sm.A*Q
        Sm.beta = Sm.beta+mutation_step_size*deltabeta
        # add velocity mutation
        Sm.vel_beta = Sm.vel_beta + vel_mutation_setp_size*random_matrix((self.k,1))
        Sm.vel_A = Sm.vel_A + vel_mutation_setp_size*random_matrix((self.d,self.k))
        return Sm

    def best_global_from(self, archive, p_gb):
        """
        assumes archive is sorted wrt crowding distance 
        """
        if np.random.random() < p_gb:
            # return the one with maximum crowding distance with prob p_gb
            return archive[-1] 
        else:
            rid = np.random.randint(0,len(archive))
            return archive[rid]

    def non_dominated_merge(self,M,archive, P):
        F = nondominated_subset(P)
        archive = nondominated_subset(F+archive)
        if len(archive) > M:
            archive = compute_crowding_distance(archive,is_nondominant=True, to_sort=True)
            archive = archive[-M:]
        return archive

    def run(self,N, M, T_max,w,c_1,c_2,eps,p_m,mutate_step_size,vel_mutation_setp_size, p_gb):
        Archive = []
        P = self.initialize_population_and_velocity(N)
        for p in P:
            _R = self.data_handler.compute_reward(p.beta,p.A)
            _L = self.data_handler.compute_loss(p.beta,p.A)
            p.set_first_RL_MOPSO(_R,_L)
        Archive = nondominated_subset(P)
        Archive = compute_crowding_distance(Archive,is_nondominant=True, to_sort=True)
        for t in tqdm(range(T_max), desc="Time loop in MOPSO"):
            best_global = self.best_global_from(Archive,p_gb)
            for i in tqdm(range(N), desc="Position update loop in MOPSO"):
                self.update_velocity(P[i], best_global, w,c_1,c_2)
                self.update_position(P[i],eps)
                if np.random.random() < p_m:
                    self.mutate(P[i],mutate_step_size, vel_mutation_setp_size)
                _R = self.data_handler.compute_reward(P[i].beta,P[i].A)
                _L = self.data_handler.compute_loss(P[i].beta,P[i].A)
                P[i].assign_RL(_R,_L)
                _temp_sol = Solution(p.beta, p.A)
                _temp_sol.assign_RL(_R,_L)
                p.update_best_pos_obj(_temp_sol)
            A = self.non_dominated_merge(M,A,P)
            A = compute_crowding_distance(A,is_nondominant=True, to_sort=True)
        