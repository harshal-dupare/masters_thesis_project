import numpy as np
from tqdm import tqdm

class Solution:
    error_eps = 1e-6
    def __init__(self):
        self.beta = None
        self.A = None

        self.R = None
        self.L = None
        self.obj = [-self.R, self.L]

        self.crowding_distance = None
        self.domination_count = None

        self.best_beta = None
        self.best_A = None
        self.best_R = None
        self.best_L = None
        self.best_obj = [-self.best_R, self.best_L]

        self.vel_A = None
        self.vel_beta = None

        pass

    def __init__(self, beta, A,vel_beta =None, vel_A=None):
        self.beta = beta
        self.A = A
        self.vel_beta = vel_beta
        self.vel_A = vel_A
        pass

    def __str__(self):
        return f"{self.R},{self.L}"

    def assign_RL(self, R, L):
        self.R = R
        self.L = L
        self.obj = [-self.R, self.L]
        pass

    def first_set_RL_NSGA(self, R, L):
        self.R = R
        self.L = L
        self.obj = [-self.R, self.L]
        pass

    def first_set_RL_MOPSO(self, R, L):
        self.R = R
        self.L = L
        self.obj = [-self.R, self.L]

        self.best_beta = self.beta
        self.best_A = self.A
        self.best_R = R
        self.best_L = L
        self.best_obj = [-self.best_R, self.best_L]
        pass

    def set_best_pos_obj_MOPSO(self, _beta, _A,_R, _L ):
        self.best_beta = _beta
        self.best_A = _A
        self.best_R = _R
        self.best_L = _L
        self.best_obj = [-self.best_R, self.best_L]
        pass
    
    def update_best_pos_obj_MOPSO(self,pos_candi):
        if pos_candi.dominates(self):
            self.set_best_pos_obj_MOPSO(pos_candi.beta,pos_candi.A,pos_candi.R,pos_candi.L)
            return True
        elif not self.dominates(pos_candi):
            if np.random.random() > 0.5:
                self.set_best_pos_obj_MOPSO(pos_candi.beta,pos_candi.A,pos_candi.R,pos_candi.L)
                return True
        return False

    def dominates(self,other):
        diff = False
        for j in range(2):
            if self.obj[j] > other.obj[j] + Solution.error_eps:
                return False
            elif self.obj[j] < other.obj[j] - Solution.error_eps:
                diff = True
        return diff
    
def nondominated_subset(P):
    n_p = len(P)
    for p in P:
        p.domination_count = 0
    cand_ids = set(range(n_p))
    for i in range(n_p):
        if i not in cand_ids:
            continue
        dom_ids = set()
        for j in cand_ids:
            if j==i:
                continue
            if P[i].dominates(P[j]):
                dom_ids.add(j)
        cand_ids.difference_update(dom_ids)
    non_dom = [P[i] for i in cand_ids]
    return non_dom

def non_dominated_sorting(P, serialize = False):
    n_p = len(P)
    for p in P:
        p.domination_count = 0
    for i in range(len(P)):
        for j in range(len(P)):
            if i!=j:
                if P[i].dominates(P[j]):
                    P[j].domination_count += 1
    P = sorted(P, key=lambda p:p.domination_count)

    F_list = [[P[0]]]
    last_dom_c, i =P[0].domination_count,1
    while i < n_p:
        if P[i].domination_count == last_dom_c:
            F_list[-1].append(P[i])
        else:
            F_list.append([P[i]])
            last_dom_c = P[i].domination_count
        i+=1
    
    F_list_return = []
    if serialize:
        for F_i in F_list:
            F_list_return += F_i
    else:
        F_list_return = F_list
    
    return F_list_return

def compute_crowding_distance(P, is_nondominant=False, serialize=False, to_sort=False):
    F_list = []
    if not is_nondominant:
        F_list = non_dominated_sorting(P)
    else:
        F_list = [P]
    
    for F_i in F_list:
        for p in F_i:
            p.crowding_distance = 0
        for j in range(2):
            F_i = sorted(F_i,key= lambda p: p.obj[j])
            for k in range(1, len(F_i)-1):
                F_i[k].crowding_distance += abs(F_i[k +1].obj[j]-F_i[k-1].obj[j])
            F_i[0].crowding_distance = np.inf
            F_i[len(F_i)-1].crowding_distance = np.inf
    
    if is_nondominant:
        if to_sort:
            F_list[0] = sorted(F_list[0], key=lambda p:p.crowding_distance)
        return F_list[0]
    else:
        if serialize:
            F_list_return = []
            for F_i in F_list:
                F_list_return+=F_i
            return F_list_return
        else:
            return F_list
