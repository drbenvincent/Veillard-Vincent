import pymc3 as pm
import numpy as np
import math


def V(reward, delay, logk):
    '''Calculate the present subjective value of a given prospect'''
    k = pm.math.exp(logk)
    return reward * discount_function(delay, k)


def discount_function(delay, k):
    ''' Hyperbolic discount function '''
    return 1 / (1.0+(k*delay))


def Φ(VA, VB, ϵ=0.01):
    '''Psychometric function which converts the decision variable (VB-VA)
    into a reponse probability. Output corresponds to probability of choosing
    the delayed reward (option B).'''
    return ϵ + (1.0-2.0*ϵ) * (1/(1+pm.math.exp(-1.7*(VB-VA))))
    

def build_model(data):
    
    RA = data['RA'].values
    RB = data['RB'].values
    DA = data['DA'].values
    DB = data['DB'].values
    R = data['R'].values
    p = data['id'].values
    n_participants = np.max(p) + 1

    with pm.Model() as model:
        '''Hierachical model with trials, participants, and groups. 
        Different (k,s) parameters for each participant. 
        Each participant comes from one of 4 groups.

        Observed data:
        - RA, DA, RB, DB, R: trial level data
        - group: list of group membership for each participant 
        - g: equals [0, 1, 2, 3] just used for group level inferences about (logk, logs)
        '''

        # Hyperpriors 
        mu_logk = pm.Normal('mu_logk', mu=math.log(1/100), sd=2)
        sigma_logk = pm.Exponential('sigma_logk', 10)


        # Priors over parameters for each participant 
        logk = pm.Normal('logk', mu=mu_logk, sd=sigma_logk, shape=n_participants) 

#         # group level inferences, unattached from the data
#         group_logk = pm.Normal('group_logk', mu=mu_logk, sd=sigma_logk) 

        # Choice function: psychometric
        P = pm.Deterministic('P', Φ(V(RA, DA, logk[p]),
                                    V(RB, DB, logk[p])) )

        # Likelihood of observations
        R = pm.Bernoulli('R', p=P, observed=R)

    return model