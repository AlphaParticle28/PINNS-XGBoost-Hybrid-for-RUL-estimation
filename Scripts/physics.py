import torch
import torch.autograd as autograd
from NN_Learner import neural, ParameterLearner

ANN = neural()
param_learner = ParameterLearner()
"""
    Arrhenius equation for the rate of a chemical reaction
    dC/dt = -k * (I ** n) * np.exp(-Ea / (R * T))
    C: Capacity (Farads)
    I: Current (Amperes)
    T: Temperature (Kelvin)
    k: Reaction rate constant (proportionality constant)
    n: Reaction order
    Ea: Activation energy (Joules)
    R: Universal gas constant (J/(mol*K))
"""
R = 8.314  
def ArrheniusRHS(T, I, params):
    k, n, Ea = params
    return -k * (I ** n) * torch.exp(-Ea / (R * T))
def ArrheniusLHS(t, T, I):
    C = ANN(t, T, I)
    dCdt = autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    return dCdt
def ArrheniusLoss(t, T, I, params):
    params = param_learner() 
    rhs = ArrheniusRHS(T, I, params)
    lhs = ArrheniusLHS(t, T, I)
    return torch.mean((lhs - rhs) ** 2)
