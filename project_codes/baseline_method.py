import numpy as np
from tqdm import tqdm
from datahandler import *
from logger import *

class Baseline_Method:
    def __init__(self, data_handler: Data_Handler):
        self.n = data_handler.n
        self.k = data_handler.k
        self.d = data_handler.d
        self.T = data_handler.T
        self.S = data_handler.S
        self.data_handler = data_handler
        pass

    def run(self, iter_max=100, logger:Logger=None):
        beta_opt = None
        A_opt = None
        R_opt = -float('inf')
        R_opt = float('inf')
        for iter in tqdm(range(iter_max), desc='Baseline Method'):
            M = random_matrix((self.d, self.k))
            A = gram_schmidt_algorithm(M)
            beta = self.data_handler.linear_l2_regression(A)
            R = self.data_handler.compute_reward(beta, A)
            L = self.data_handler.compute_loss(beta, A)
            if logger is not None:
              logger.log_RL(R,L)
            if R > R_opt:
                beta_opt = beta
                A_opt = A
                R_opt = R
                L_opt = L
        return beta_opt, A_opt
