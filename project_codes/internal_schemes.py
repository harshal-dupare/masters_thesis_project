import numpy as np
from tqdm import tqdm
from datahandler import *

class U_A_Method:
    def __init__(self):
        pass
class U_A_Method_1(U_A_Method):
    def __init__(self, alpha, data_handler:Data_Handler):
        super().__init__()
        self.data_handler = data_handler
        self.itter = alpha[0]
        self.learning_rate = alpha[1]
        self.weight_R = alpha[2]
        self.weight_L = alpha[3]
        self.weight_O = alpha[4]
        pass
    def get_delta_A(self, beta, A):
        R_grad_A = self.data_handler.compute_reward_gradient_A(beta, A)
        L_grad_A = self.data_handler.compute_loss_gradient_A(beta, A)
        O_grad_A = self.data_handler.compute_orthogonal_condition_normF_gradient_A(A)
        delta = self.weight_R*R_grad_A-self.weight_L*L_grad_A-self.weight_O*O_grad_A
        return self.learning_rate*delta
    def get_string(self):
        return f"U_A[{self.itter},{int(10000*self.learning_rate)},{int(10000*self.weight_R)},{int(10000*self.weight_L)},{int(10000*self.weight_O)}]"


class U_beta_Method:
    def __init__(self):
        pass
class U_beta_Method_1(U_beta_Method):
    def __init__(self, phi, data_handler:Data_Handler):
        super().__init__()
        self.data_handler = data_handler
        self.itter = phi[0]
        self.learning_rate = phi[1]
        self.weight_R = phi[2]
        self.weight_L = phi[3]
        pass
    def get_delta_beta(self, beta, A, A_new):
        R_grad_beta = self.data_handler.compute_reward_gradient_beta(beta, A)
        L_grad_beta = self.data_handler.compute_loss_gradient_beta(beta, A)
        delta = self.weight_R*R_grad_beta-self.weight_L*L_grad_beta
        return self.learning_rate*delta
    def get_string(self):
        return f"U_beta[{self.itter},{int(10000*self.learning_rate)},{int(10000*self.weight_R)},{int(10000*self.weight_L)}]"
class U_beta_Method_2(U_beta_Method):
    def __init__(self, phi, data_handler:Data_Handler):
        super().__init__()
        self.data_handler = data_handler
        self.itter = phi[0]
        self.learning_rate = phi[1]
        self.weight_R = phi[2]
        self.weight_L = phi[3]
        self.weight_reg = phi[4]
        pass
    def get_delta_beta(self, beta, A, A_new):
        R_grad_beta = self.data_handler.compute_reward_gradient_beta(beta, A)
        L_grad_beta = self.data_handler.compute_loss_gradient_beta(beta, A)
        beta_reg = self.data_handler.linear_l2_regression(A_new)

        delta = self.weight_R*R_grad_beta-self.weight_L*L_grad_beta
        delta = (1-self.weight_reg)*self.learning_rate*delta
        delta += self.weight_reg*(beta_reg-2*beta) 
        return delta
    def get_string(self):
        return f"U_beta[{self.itter},{int(10000*self.learning_rate)},{int(10000*self.weight_R)},{int(10000*self.weight_L)},{int(10000*self.weight_reg)}]"

class U_O_Method:
    def __init__(self):
        pass
class U_O_Method_1(U_O_Method):
    def __init__(self, lamb, data_handler:Data_Handler):
        super().__init__()
        self.data_handler = data_handler
        self.iter_max = lamb[0]
        self.learning_rate_A = lamb[1]
        self.learning_rate_beta = lamb[2]
        self.weight_reg = lamb[3]
        pass
    def get_new_beta_A(self, beta, A):
        A_i = random_matrix(A.shape)
        A_i = gram_schmidt_algorithm(A_i)
        beta_i = self.data_handler.linear_l2_regression(A_i)
        A_opt = A_i
        beta_opt = beta_i
        R_opt = self.data_handler.compute_reward(beta_opt,A_opt)
        L_opt = self.data_handler.compute_loss(beta_opt,A_opt)
        for iter in tqdm(range(self.iter_max),desc="U_O_Method_1:get_new_beta_A"):
            deltaA = self.learning_rate_A *(A - A_i)
            T_i =  0.5*(np.matmul(deltaA.T,A_i) - np.matmul(A_i.T,deltaA))
            Q = cayley_transformation(T_i)
            A_i = np.matmul(A_i, Q)
            beta_reg = self.data_handler.linear_l2_regression(A_i)
            beta_grad = self.data_handler.compute_reward_gradient_beta(beta_i, A_i)
            beta_i = beta_i + self.learning_rate_beta*((1-self.weight_reg)*beta_grad+self.weight_reg*(beta_reg-beta_i))
            R_i = self.data_handler.compute_reward(beta_i,A_i)
            L_i = self.data_handler.compute_loss(beta_i,A_i)
            if R_i > R_opt:
                A_opt = A_i
                beta_opt = beta_i
                R_opt = R_i
        return beta_opt, A_opt
    def get_string(self):
        return f"U_O[{self.iter_max},{int(10000*self.learning_rate_A)},{int(10000*self.learning_rate_beta)},{int(10000*self.weight_reg)}]"