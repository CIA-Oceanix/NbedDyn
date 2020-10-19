from code import *
import torch
from torch.autograd.gradcheck import zero_gradients
import numpy as np
from torch.autograd import Variable
from scipy.stats import multivariate_normal
from tqdm import tqdm
from scipy.integrate import odeint
from numpy.linalg import pinv

def RMSE(a,b):
    """ Compute the Root Mean Square Error between 2 n-dimensional vectors. """
 
    return np.sqrt(np.mean((a-b)**2))

def Mean_RMSE(a,b):
    """ Compute the Root Mean Square Error between 2 n-dimensional vectors. """
    if (a.ndim==1):
        a = a[np.newaxis]
    if (a.ndim>2):
        a = a.reshape(a.shape[0],-1)
    if (b.ndim==1):
        b = b[np.newaxis]    
    if (b.ndim>2):
        b = b.reshape(b.shape[0],-1)
    return np.sqrt(np.nanmean((a-b)**2,1))
    
    
def hanning2d(M, N):
    """
    A 2D hanning window, as per IDL's hanning function.  See numpy.hanning for the 1d description
    """
    
    if N <= 1:
        return np.hanning(M)
    elif M <= 1:
        return np.hanning(N) # scalar unity; don't window if dims are too small
    else:
        return np.outer(np.hanning(M),np.hanning(N))

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi, rho)

def Corr(a,b):
    """ Compute the Correlation between 2 n-dimensional vectors. """
    if (a.ndim==1):
        a = a[np.newaxis]
    if (a.ndim>2):
        a = a.reshape(a.shape[0],-1)
    if (b.ndim==1):
        b = b[np.newaxis] 
    if (b.ndim>2):
        b = b.reshape(b.shape[0],-1)
    a = a - np.nanmean(a,1)[np.newaxis].T
    b = b - np.nanmean(b,1)[np.newaxis].T
    r = np.nansum((a*b),1) / np.sqrt(np.nansum((a*a),1) * np.nansum((b*b),1))
    return r 

def raPsd2dv1(img,res,hanning):
    """ Computes and plots radially averaged power spectral density (power
     spectrum) of image IMG with spatial resolution RES.
    """
    
    img = img.copy()
    N, M = img.shape
    if hanning:
        img = hanning2d(*img.shape) * img        
    imgf = np.fft.fftshift(np.fft.fft2(img))
    imgfp = np.power(np.abs(imgf)/(N*M),2)    
    # Adjust PSD size
    dimDiff = np.abs(N-M)
    dimMax = max(N,M)
    if (N>M):
        if ((dimDiff%2)==0):
            imgfp = np.pad(imgfp,((0,0),(dimDiff/2,dimDiff/2)),'constant',constant_values=np.nan)
        else:
            imgfp = np.pad(imgfp,((0,0),(dimDiff/2,1+dimDiff/2)),'constant',constant_values=np.nan)
            
    elif (N<M):
        if ((dimDiff%2)==0):
            imgfp = np.pad(imgfp,((dimDiff/2,dimDiff/2),(0,0)),'constant',constant_values=np.nan)
        else:
            imgfp = np.pad(imgfp,((dimDiff/2,1+dimDiff/2),(0,0)),'constant',constant_values=np.nan)
    halfDim = int(np.ceil(dimMax/2.))
    X, Y = np.meshgrid(np.arange(-dimMax/2.,dimMax/2.-1+0.00001),np.arange(-dimMax/2.,dimMax/2.-1+0.00001))           
    theta, rho = cart2pol(X, Y)                                              
    rho = np.round(rho+0.5)   
    Pf = np.zeros(halfDim)
    f1 = np.zeros(halfDim)
    for r in range(halfDim):
      Pf[r] = np.nansum(imgfp[rho == (r+1)])
      f1[r] = float(r+1)/dimMax
    f1 = f1/res
    return f1, Pf
def compute_jacobian(inputs, output):
    assert inputs.requires_grad
    num_classes = output.size()[1]
    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()
    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data
    return torch.transpose(jacobian, dim0=0, dim1=1)
def proj(u, v):
    # notice: this algrithm assume denominator isn't zero
    return u * np.dot(v,u) / np.dot(u,u)  

def GS(V):
    V = 1.0 * V     # to float
    U = np.copy(V)
    for i in range(1, V.shape[1]):
        for j in range(i):
            U[:,i] -= proj(U[:,j], V[:,i])
    # normalize column
    den=(U**2).sum(axis=0) **0.5
    E = U/den
    # assert np.allclose(E.T, np.linalg.inv(E))
    return U,den,E

def Compute_Lyapunov_spectrum(dyn_mdl, init_state, pred_time_steps, init_cov_factor, dt_integration, need_timestep):
    #args : dynamical model, initial state, number of prediction timesteps, init cov factor
    #returns : lyapunov exponents and lyapunov dimension
    #
    tmp = np.reshape(init_state,(1,len(init_state)))
    cov_init = init_cov_factor*np.eye(len(init_state))
    z = np.zeros((pred_time_steps,len(init_state),len(init_state)))
    y = np.zeros((pred_time_steps,len(init_state),len(init_state)))
    l = np.zeros((pred_time_steps,len(init_state)))
    w = np.zeros((pred_time_steps,len(init_state),len(init_state)))
    cov = np.zeros((pred_time_steps,len(init_state),len(init_state)))
    l[0,:] = np.ones((len(init_state)))
    w[0,:,:]=cov_init
    cov[0,:,:]=cov_init

    for i in range(1,pred_time_steps):
        tmp = Variable(torch.from_numpy(tmp).float())
        tmp.requires_grad = True
        if need_timestep == True: 
            tmp_out = dyn_mdl(tmp,dt_integration)[0]
        else:
            tmp_out = dyn_mdl(tmp)
        tmp_out = tmp_out.reshape((1,len(init_state)))
        jac = compute_jacobian(tmp, tmp_out)
        z[i,:,:] = np.dot(jac,w[i-1,:,:])[0]
        y[i,:,:],l[i,:],w[i,:,:]=GS(z[i,:,:])
        tmp = np.reshape(tmp_out.data.numpy(),(1,len(init_state)))
        cov[i,:,:] = np.dot(np.dot(jac.data.numpy()[0,:,:],w[i-1,:,:]),jac.data.numpy()[0,:,:].T)

    l_exp = np.sum(np.log(l[2:,:]),axis = 0)/(np.shape(l[2:,:])[0]-1)/dt_integration
    l_dim = 2+(1./np.abs(l_exp[-1]))*np.sum(l_exp[:2])
    return l_exp, l_dim


def compute_largest_Lyapunov(modelRINN, x0,dt_integration,d0,nb_iter, need_timestep):
    d0 = d0*np.ones(x0.shape[-1])
    forecasted_states, forecasted_states_noisy, forecasted_states_noisy_proj, log_pret = [],[],[],[]
    tmp = np.reshape(x0,(1,x0.shape[-1]))
    
    tmp_init = Variable(torch.from_numpy(np.reshape(tmp,(1,x0.shape[-1]))).float())
    tmp_noisy = Variable(torch.from_numpy(np.reshape(tmp + d0,(1,x0.shape[-1]))).float())
    lyap_series = []
    for i in range(nb_iter):
        #1 - forecast the states :
        if need_timestep == True: 
            forecasted_states.append(modelRINN(tmp_init,dt_integration)[0][0,:].data.numpy())
            forecasted_states_noisy.append(modelRINN(tmp_noisy,dt_integration)[0][0,:].data.numpy())
        
        else:
            forecasted_states.append(modelRINN(tmp_init)[0,:].data.numpy())
            forecasted_states_noisy.append(modelRINN(tmp_noisy)[0,:].data.numpy())
        d1 = np.linalg.norm(forecasted_states_noisy[-1]-forecasted_states[-1])
        #compute log(d1/d0)
        log_pret.append(np.log(d1/d0))
        
        # readjusting orbits
        forecasted_states_noisy_proj.append(forecasted_states[-1]+d0*(forecasted_states_noisy[-1]-forecasted_states[-1])/d1)
        tmp_init = np.reshape(forecasted_states[-1],(1,x0.shape[-1]))
        tmp_noisy = np.reshape(forecasted_states_noisy_proj[-1],(1,x0.shape[-1]))
        
        tmp_init = Variable(torch.from_numpy(tmp_init).float())
        tmp_noisy = Variable(torch.from_numpy(tmp_noisy).float())
        
        lyap_series.append(np.mean(log_pret)/dt_integration)
    return lyap_series, forecasted_states