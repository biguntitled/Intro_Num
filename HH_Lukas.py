from types import SimpleNamespace
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt



class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

        #g. initial guess for allocation
        par.guess = [4.5, 4.5, 4.4, 4.4]

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol    

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if np.isclose(par.sigma,0):
            H = min(HF, HM)
        elif np.isclose(par.sigma,1): 
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            power1 = (par.sigma-1)/par.sigma
            power2 = par.sigma/(par.sigma-1)
            H = ((1-par.alpha)*HM**(power1)+par.alpha*HF**(power1))**(power2)

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve_continuous(self, do_print=False):
        """ solve model continuously """
    
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. set bounds
        bounds = [(0, 24), (0, 24), (0, 24), (0, 24)]

        # b. find maximizing argument
        res = optimize.minimize(lambda x: -self.calc_utility(*x),method='SLSQP', x0=par.guess, bounds=bounds)
        
        opt.LM = res.x[0]
        opt.HM = res.x[1]
        opt.LF = res.x[2]
        opt.HF = res.x[3]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        sol = self.sol
        res = SimpleNamespace()
        wF = self.par.wF_vec
        it = len(wF)
        Home_ratios = np.zeros(it)
        LM = np.zeros(it)
        HM = np.zeros(it)
        LF = np.zeros(it)
        HF = np.zeros(it)
        if discrete == True:
            for it, val in enumerate(wF): 
                self.par.wF = val
                opt = self.solve_discrete()
                Home_ratios[it] = opt.HF / opt.HM ##Store the ratio
                LM[it]= opt.LM #But also the individual values
                HM[it]= opt.HM
                LF[it]= opt.LF
                HF[it]= opt.HF
                res.Home_ratios = Home_ratios
                res.LM = LM
                res.HM = HM
                res.LF = LF
                res.HF = HF
                sol.LM_vec = LM #Store globally
                sol.HM_vec = HM
                sol.LF_vec = LF
                sol.HF_vec = HF
            return res
        else: 
            for it, val in enumerate(wF): 
                self.par.wF = val
                opt = self.solve_continuous()
                Home_ratios[it] = opt.HF / opt.HM ##Store the ratio
                LM[it]= opt.LM #But also the individual values
                HM[it]= opt.HM
                LF[it]= opt.LF
                HF[it]= opt.HF
                res.Home_ratios = Home_ratios
                res.LM = LM
                res.HM = HM
                res.LF = LF
                res.HF = HF
                sol.LM_vec = LM #Store globally
                sol.HM_vec = HM
                sol.LF_vec = LF
                sol.HF_vec = HF
            return res
    

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        pass