from code import *
import numpy as np

def Lorenz_63(S,t,sigma,rho,beta):
    """ Lorenz-63 dynamical model. """
    x_1 = sigma*(S[1]-S[0]);
    x_2 = S[0]*(rho-S[2])-S[1];
    x_3 = S[0]*S[1] - beta*S[2];
    dS  = np.array([x_1,x_2,x_3]);
    return dS

def Lorenz_96(S,t,F,J):
    """ Lorenz-96 dynamical model. """
    x = np.zeros(J);
    x[0] = (S[1]-S[J-2])*S[J-1]-S[0];
    x[1] = (S[2]-S[J-1])*S[0]-S[1];
    x[J-1] = (S[0]-S[J-3])*S[J-2]-S[J-1];
    for j in range(2,J-1):
        x[j] = (S[j+1]-S[j-2])*S[j-1]-S[j];
    dS = x.T + F;
    return dS


def oregonator(S,t,alpha,beta,sigma):
    """ Lorenz-63 dynamical model. """
    x_1 = alpha*(S[1]+S[0]*(1-beta*(1e-6)*S[0]-S[1]));
    x_2 = (1/alpha)*(S[2]-(1+S[0])*S[1]);
    x_3 = sigma*(S[0]-S[2]);
    dS  = np.array([x_1,x_2,x_3]);
    return dS

def Adv_Dif_1D(t,S,w):
    """ Adv/Dif/1D dynamical model. """
    x_1 = w*(S);
    dS  = np.array([x_1]);
    return dS

# define here your own dynamical model
