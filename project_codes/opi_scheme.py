import numpy as np
from tqdm import tqdm
from internal_schemes import *
from datahandler import *
from logger import *

class OPI:
    def __init__(self,data_handler:Data_Handler, U_beta:U_beta_Method, U_A:U_A_Method):
        self.n = data_handler.n
        self.k = data_handler.k
        self.d = data_handler.d
        self.T = data_handler.T
        self.S = data_handler.S
        self.data_handler = data_handler
        self.U_beta = U_beta
        self.U_A = U_A
        pass

    def run(self, iter_max=100, logger:Logger=None):
        A = random_matrix((self.d, self.k))
        A = gram_schmidt_algorithm(A)
        beta = self.data_handler.linear_l2_regression(A)
        for iter in tqdm(range(iter_max), desc='OPI Scheme'):
            deltaA = self.U_A.get_delta_A(beta,A)
            T_i =  0.5*(np.matmul(deltaA.T,A) - np.matmul(A.T,deltaA))
            Q = cayley_transformation(T_i)
            AQ = np.matmul(A,Q)
            beta = beta + self.U_beta.get_delta_beta(beta,A,AQ)
            A = AQ
            if logger is not None:
                logger.log_RL(self.data_handler.compute_reward(beta,A),self.data_handler.compute_loss(beta,A))
        return beta, A
        
