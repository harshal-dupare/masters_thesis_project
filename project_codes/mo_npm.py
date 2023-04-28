import numpy as np
from tqdm import tqdm
from datahandler import *
from logger import *

class MO_NPM_1:
    def __init__(self, data_handler: Data_Handler, learning_rate_A,learning_rate_beta):
        self.n = data_handler.n
        self.k = data_handler.k
        self.d = data_handler.d
        self.T = data_handler.T
        self.S = data_handler.S
        self.data_handler = data_handler
        self.learning_rate_A = learning_rate_A
        self.learning_rate_beta = learning_rate_beta
        pass

    def run(self, iter_max=100, logger:Logger=None):
        A_i = random_matrix((self.d,self.k))
        A_i = gram_schmidt_algorithm(A_i)
        beta_i = self.data_handler.linear_l2_regression(A_i)
        R_i = self.data_handler.compute_reward(beta_i, A_i)
        L_i = self.data_handler.compute_loss(beta_i, A_i)
        A_opt = A_i
        beta_opt = beta_i
        R_opt = R_i
        L_opt = L_i
        for iter in tqdm(range(iter_max), desc="MO_NPM_2"):
            R_grad_beta, R_grad_A = self.data_handler.compute_reward_gradient_beta_A(beta_i, A_i)
            L_grad_beta, L_grad_A = self.data_handler.compute_loss_gradient_beta_A(beta_i, A_i)
            O_grad_A = self.data_handler.compute_orthogonal_condition_normF_gradient_A(A_i)
            deltaA = float(1-R_i)*R_grad_A - float(L_i)*L_grad_A - 0.5*O_grad_A
            deltabeta = float(1-R_i)*R_grad_beta - float(L_i)*L_grad_beta
            T_i =  0.5*self.learning_rate_A *(np.matmul(deltaA.T,A_i) - np.matmul(A_i.T,deltaA))
            Q = cayley_transformation(T_i)
            A_i = np.matmul(A_i, Q)
            beta_i = beta_i + self.learning_rate_beta*deltabeta
            R_i = self.data_handler.compute_reward(beta_i, A_i)
            L_i = self.data_handler.compute_loss(beta_i, A_i)
            if logger is not None:
                logger.log_RLO(R_i,L_i,self.data_handler.compute_orthogonal_condition_normF(A_i))
            if R_i > R_opt:
                A_opt = A_i
                beta_opt = beta_i
                R_opt = R_i
                L_opt = L_i
        return beta_opt, A_opt

    def get_string(self):
        return f"NPM1[{int(1000*self.learning_rate_A)}-{int(1000*self.learning_rate_beta)}]"

class MO_NPM_2:
    def __init__(self, data_handler: Data_Handler,learning_rate_A, learning_rate_beta):
        self.n = data_handler.n
        self.k = data_handler.k
        self.d = data_handler.d
        self.T = data_handler.T
        self.S = data_handler.S
        self.data_handler = data_handler
        self.learning_rate_A = learning_rate_A
        self.learning_rate_beta = learning_rate_beta
        pass

    def run(self, iter_max=100, logger:Logger=None):
        A_i = random_matrix((self.d,self.k))
        A_i = gram_schmidt_algorithm(A_i)
        beta_i = self.data_handler.linear_l2_regression(A_i)
        R_i = self.data_handler.compute_reward(beta_i, A_i)
        L_i = self.data_handler.compute_loss(beta_i, A_i)
        A_opt = A_i
        beta_opt = beta_i
        R_opt = R_i
        L_opt = L_i
        for iter in tqdm(range(iter_max), desc="MO_NPM_2"):
            R_grad_beta, R_grad_A = self.data_handler.compute_reward_gradient_beta_A(beta_i, A_i)
            L_grad_beta, L_grad_A = self.data_handler.compute_loss_gradient_beta_A(beta_i, A_i)
            O_grad_A = self.data_handler.compute_orthogonal_condition_normF_gradient_A(A_i)
            deltaA = float(1-R_i)*R_grad_A - float(L_i)*L_grad_A - 0.5*O_grad_A
            deltabeta = float(1-R_i)*R_grad_beta - float(L_i)*L_grad_beta
            A_i = A_i + self.learning_rate_A * deltaA
            beta_i = beta_i + self.learning_rate_beta*deltabeta
            R_i = self.data_handler.compute_reward(beta_i, A_i)
            L_i = self.data_handler.compute_loss(beta_i, A_i)
            if logger is not None:
                logger.log_RLO(R_i,L_i,self.data_handler.compute_orthogonal_condition_normF(A_i))
            if R_i > R_opt:
                A_opt = A_i
                beta_opt = beta_i
                R_opt = R_i
                L_opt = L_i
        return beta_opt, A_opt
        
    def get_string(self):
        return f"NPM2[{int(1000*self.learning_rate_A)}-{int(1000*self.learning_rate_beta)}]"
