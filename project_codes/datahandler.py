import numpy as np

def gram_schmidt_algorithm(A):
    """
    O(d^2k)
    """
    for i in range(A.shape[0]):
        q = A[i, :]
        for j in range(i):
            q = q - np.dot(A[j,:], A[i,:]) * A[j,:]
        q = q / np.sqrt(np.dot(q, q))
        A[i,:] = q
    return A

def cayley_transformation(A):
    """
    A is (d,k) matrix
    T = 
    O()
    """
    I = np.eye(A.shape[0])
    Q = np.matmul(np.linalg.inv(I+A),(I-A))
    return Q

def random_matrix(shape, limits_gap=200, center=0.5):
    rmat = np.random.random(shape)
    while np.linalg.matrix_rank(rmat) < min(shape[0],shape[1]):
        rmat = np.random.random(shape)
    return limits_gap*(rmat-center)

class Data_Handler:
    def __init__(self, n, k, d, T, S, R):
        """
        R is a n x (T + S) numpy array
        d is an integer that represents the number of time lags
        d <= k
        """
        self.n = n
        self.k = k
        self.d = d
        self.T = T
        self.S = S
        self.R = R
        assert(d<=k)
        assert(R.shape[0]==n)
        assert(R.shape[1]==(T+S))
        self._compute_optimizers()
        pass

    def get_string(self):
        return f"{self.n}-{self.k}-{self.d}-{self.T}-{self.S}"

    def _compute_optimizers(self):
        self.sum_RtdTRtd_dT_altD = np.zeros((self.d, self.d))  # d x d
        self.sum_RtdTRt_dT_altN = np.zeros((self.d, 1))  # d x 1
        self.list_RtdTRtd = []
        self.list_RtdTRt = []
        self.list_norm_Rt = []
        for t in range(self.d):
            r_t = self.R[:, t].reshape(self.n, 1)  # n x 1
            self.list_RtdTRtd += [None]
            self.list_RtdTRt += [None]
            self.list_norm_Rt += [np.linalg.norm(r_t, 2)]
        for t in range(self.d, self.T+self.S):
            r_t = self.R[:, t].reshape(self.n, 1)  # n x 1
            r_td = self.R[:, t-self.d:t]  # n x d
            self.list_RtdTRtd += [np.matmul(r_td.T, r_td)]
            self.list_RtdTRt += [np.matmul(r_td.T, r_t)]
            self.list_norm_Rt += [np.linalg.norm(r_t, 2)]
        for t in range(self.d, self.T):
            self.sum_RtdTRtd_dT_altD += self.list_RtdTRtd[t]
            self.sum_RtdTRt_dT_altN += self.list_RtdTRt[t]

    def compute_reward(self, beta, A):
        Abeta = np.matmul(A, beta)  # d x 1
        reward_sum = 0
        for t in range(self.T, self.T + self.S):
            reward_sum += np.matmul(self.list_RtdTRt[t].T, Abeta) / (self.list_norm_Rt[t] * np.sqrt(np.matmul(Abeta.T, np.matmul(self.list_RtdTRtd[t], Abeta))))
        reward = reward_sum / self.S
        return reward

    def compute_loss(self, beta, A):
        Abeta = np.matmul(A, beta)  # d x 1
        loss_sum = 0
        for t in range(self.d, self.T):
            loss_sum += self.list_norm_Rt[t]
            loss_sum += -2*np.matmul(self.list_RtdTRt[t].T, Abeta)
            loss_sum += np.matmul(Abeta.T, np.matmul(self.list_RtdTRtd[t], Abeta))
        loss = loss_sum / (2*(self.T - self.d + 1))
        return loss

    def compute_loss_gradient_beta(self, beta, A):
        beta_grad = np.matmul(A.T, np.matmul(self.sum_RtdTRtd_dT_altD, np.matmul(A, beta))) - np.matmul(A.T, self.sum_RtdTRt_dT_altN)
        beta_grad = beta_grad / (self.T - self.d + 1)
        return beta_grad

    def compute_loss_gradient_A(self, beta, A):
        A_grad = np.matmul(self.sum_RtdTRtd_dT_altD, np.matmul(A, beta)) - self.sum_RtdTRt_dT_altN
        A_grad = np.matmul(A_grad, beta.T)
        A_grad = A_grad / (self.T - self.d + 1)
        return A_grad

    def compute_loss_gradient_beta_A(self, beta, A):
        temp = np.matmul(self.sum_RtdTRtd_dT_altD, np.matmul(A, beta))
        beta_grad = np.matmul(A.T, temp) - np.matmul(A.T, self.sum_RtdTRt_dT_altN)
        beta_grad = beta_grad / (self.T - self.d + 1)
        A_grad = temp - self.sum_RtdTRt_dT_altN
        A_grad = np.matmul(A_grad, beta.T)
        A_grad = A_grad / (self.T - self.d + 1)
        return beta_grad, A_grad

    def compute_reward_gradient_beta(self, beta, A):
        T1 = np.zeros((self.d,1))
        T2 = np.zeros((self.d,self.d))
        Abeta = np.matmul(A, beta)  # d x 1
        for t in range(self.T, self.T + self.S):
            Ftbeta_norm = float(np.sqrt(np.matmul(Abeta.T,np.matmul(self.list_RtdTRtd[t],Abeta))))
            RtFtbeta_scalar = float(np.matmul(self.list_RtdTRt[t].T, Abeta))
            T1 += self.list_RtdTRt[t]/(Ftbeta_norm * self.list_norm_Rt[t])
            T2 += (RtFtbeta_scalar/(self.list_norm_Rt[t]*Ftbeta_norm**3)  * self.list_RtdTRtd[t])
        grad_beta = np.matmul(A.T, T1) - np.matmul(A.T, np.matmul(T2,Abeta))
        grad_beta =  grad_beta / self.S
        return grad_beta

    def compute_reward_gradient_A(self, beta, A):
        T1 = np.zeros((self.d,1))
        T2 = np.zeros((self.d,self.d))
        Abeta = np.matmul(A, beta)  # d x 1
        for t in range(self.T, self.T + self.S):
            Ftbeta_norm = float(np.sqrt(np.matmul(Abeta.T,np.matmul(self.list_RtdTRtd[t],Abeta))))
            RtFtbeta_scalar = float(np.matmul(self.list_RtdTRt[t].T, Abeta))
            T1 += self.list_RtdTRt[t]/(Ftbeta_norm * self.list_norm_Rt[t])
            T2 += (RtFtbeta_scalar/(self.list_norm_Rt[t]*Ftbeta_norm**3)  * self.list_RtdTRtd[t])
        grad_A = np.matmul(T1, beta.T) - np.matmul(np.matmul(T2, Abeta), beta.T)
        grad_A =  grad_A / self.S
        return grad_A
    
    def compute_reward_gradient_beta_A(self, beta, A):
        T1 = np.zeros((self.d,1))
        T2 = np.zeros((self.d,self.d))
        Abeta = np.matmul(A, beta)  # d x 1
        for t in range(self.T, self.T + self.S):
            Ftbeta_norm = float(np.sqrt(np.matmul(Abeta.T,np.matmul(self.list_RtdTRtd[t],Abeta))))
            RtFtbeta_scalar = float(np.matmul(self.list_RtdTRt[t].T, Abeta))
            T1 += self.list_RtdTRt[t]/(Ftbeta_norm * self.list_norm_Rt[t])
            T2 += (RtFtbeta_scalar/(self.list_norm_Rt[t]*Ftbeta_norm**3)  * self.list_RtdTRtd[t])
        grad_beta = np.matmul(A.T, T1) - np.matmul(A.T, np.matmul(T2,Abeta))
        grad_beta =  grad_beta / self.S
        grad_A = np.matmul(T1, beta.T) - np.matmul(np.matmul(T2, Abeta), beta.T)
        grad_A =  grad_A / self.S
        return grad_beta, grad_A
    
    def compute_orthogonal_condition_normF(self, A):
        return np.sum((np.matmul(A, A.T) - np.eye(A.shape[0]))**2)

    def compute_orthogonal_condition_normF_gradient_A(self, A):
        return -4.0*np.matmul(np.eye(A.shape[0]) - np.matmul(A, A.T), A)

    def linear_l2_regression(self, A):
        """
        O(d^2k+k^3) = O(k^3)
        """
        beta = np.matmul(np.matmul(A.T,self.sum_RtdTRtd_dT_altD),A)
        beta = np.linalg.inv(beta)
        beta = np.matmul(beta, np.matmul(A.T,self.sum_RtdTRt_dT_altN))
        return beta


