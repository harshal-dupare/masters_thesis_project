import numpy as np
from tqdm import tqdm
from internal_schemes import *
from datahandler import *
from logger import *


class DOI:
    def __init__(self, data_handler: Data_Handler, U_beta: U_beta_Method, U_A: U_A_Method, U_O: U_O_Method):
        self.n = data_handler.n
        self.k = data_handler.k
        self.d = data_handler.d
        self.T = data_handler.T
        self.S = data_handler.S
        self.data_handler = data_handler
        self.U_beta = U_beta
        self.U_A = U_A
        self.U_O = U_O
        pass

    def run(self, iter_max=100, logger:Logger=None):
        A = random_matrix((self.d, self.k),2)
        beta = self.data_handler.linear_l2_regression(A)
        for iter in tqdm(range(iter_max), desc='DOI Scheme'):
            A_new = A + self.U_A.get_delta_A(beta, A)
            beta = self.U_beta.get_delta_beta(beta, A, A_new)
            A = A_new
            if logger is not None:
                logger.log_RL(self.data_handler.compute_reward(beta,A),self.data_handler.compute_loss(beta,A))
        beta, A = self.U_O.get_new_beta_A(beta, A)
        return beta, A
