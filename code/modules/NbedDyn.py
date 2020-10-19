from code import *
import numpy as np
import torch
def get_NbedDyn_model(params):
        np.random.seed(params['seed'])
        torch.manual_seed(params['seed'])
        class FC_net(torch.nn.Module):
                    def __init__(self, params):
                        super(FC_net, self).__init__()
                        y_aug = np.random.uniform(size=(params['nb_batch'],params['Batch_size'],params['dim_latent']))-0.5
                        y_aug[:,:,1:] = 0.0
                        self.y_aug = torch.nn.Parameter(torch.from_numpy(y_aug).float())
                        self.linearCell   = torch.nn.Linear(params['dim_latent']+params['dim_observations'], params['dim_hidden_linear']) 
                        self.BlinearCell1 = torch.nn.ModuleList([torch.nn.Linear(params['dim_latent']+params['dim_observations'], 1,bias = False) for i in range(params['bi_linear_layers'])])
                        self.BlinearCell2 = torch.nn.ModuleList([torch.nn.Linear(params['dim_latent']+params['dim_observations'], 1,bias = False) for i in range(params['bi_linear_layers'])])
                        augmented_size    = params['bi_linear_layers'] + params['dim_hidden_linear']
                        self.transLayers = torch.nn.ModuleList([torch.nn.Linear(augmented_size, params['dim_latent']+params['dim_observations'])])
                        self.transLayers.extend([torch.nn.Linear(params['dim_latent']+params['dim_observations'], params['dim_latent']+params['dim_observations']) for i in range(1, params['transition_layers'])])
                        #self.outputLayer  = torch.nn.Linear(params['dim_latent']+params['dim_input'], params['dim_latent']+params['dim_input'],bias = False) 
                    def forward(self, inp, dt):
                        """
                        In the forward function we accept a Tensor of input data and we must return
                        a Tensor of output data. We can use Modules defined in the constructor as
                        well as arbitrary operators on Tensors.
                        """
                        if inp.shape[-1]<params['dim_latent']+params['dim_observations']:
                            aug_inp = torch.cat((inp, self.y_aug), dim=1)
                        else:
                            aug_inp = inp
                        BP_outp = (torch.zeros((aug_inp.size()[0],params['bi_linear_layers'])))
                        L_outp   = self.linearCell(aug_inp)
                        for i in range((params['bi_linear_layers'])):
                            BP_outp[:,i]=self.BlinearCell1[i](aug_inp)[:,0]*self.BlinearCell2[i](aug_inp)[:,0]
                        aug_vect = torch.cat((L_outp, BP_outp), dim=1)
                        for i in range((params['transition_layers'])):
                            aug_vect = (self.transLayers[i](aug_vect))
                        grad = aug_vect#self.outputLayer(aug_vect)
                        return grad, aug_inp
        model  = FC_net(params)
        
        class INT_net(torch.nn.Module):
                def __init__(self, params):
                    super(INT_net, self).__init__()
        #            self.add_module('Dyn_net',FC_net(params))
                    self.Dyn_net = model
                def forward(self, inp, dt):
                        k1, aug_inp   = self.Dyn_net(inp,dt)
                        inp_k2 = inp + 0.5*dt*k1
                        k2, tmp   = self.Dyn_net(inp_k2,dt)
                        inp_k3 = inp + 0.5*dt*k2       
                        k3, tmp   = self.Dyn_net(inp_k3,dt)
                        inp_k4 = inp + dt*k3          
                        k4, tmp   = self.Dyn_net(inp_k4,dt)            
                        pred = aug_inp +dt*(k1+2*k2+2*k3+k4)/6 
                        return pred, k1, inp, aug_inp
        modelRINN = INT_net(params)
        return model, modelRINN
def train_NbedDyn_model_L63(params,model,modelRINN,X_train,Grad_t):    
        dt = params['dt_integration']
        aug_inp_data = []
        x = torch.from_numpy(X_train).float()
        z = torch.from_numpy(Grad_t).float()
        
        
        criterion = torch.nn.MSELoss(reduction='elementwise_mean')
        
        if params['pretrained'] :
            modelRINN.load_state_dict(torch.load(path + file_name +'.pt'))
        optimizer = torch.optim.Adam(model.parameters())
        
        for param_group in optimizer.param_groups:
                print(param_group['lr'])
        for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1      
        
        for t in range(0,params['ntrain'][0]):
            for b in range(params['nb_batch']):
                # Forward pass: Compute predicted y by passing x to the model
                inp_concat = torch.cat((x[b,:,:], modelRINN.Dyn_net.y_aug[b,:,:]), dim=1)
                pred, grad, inp, aug_inp     = modelRINN(inp_concat,dt)
                pred2, grad2, inp2, aug_inp2 = modelRINN(pred,dt)
                if params['get_latent_train']:
                    aug_inp_data.append(aug_inp.detach())
                # Compute and print loss
                loss1 = criterion(grad[:,:1], z[b,:,:])
                loss2 = criterion(pred[:-1,1:] , aug_inp[1:,1:])
                loss3 = criterion(pred2[:-1,1:] , pred[1:,1:])
                loss =  0.1*loss1+0.9*loss2 + 0.9*loss3
                if t%1000==0:
                    print('Training L63 NbedDyn model', t,loss)
                torch.save(modelRINN.state_dict(), params['path'] + params['file_name']+'.pt')
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            if t>1500:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.01
            if t>5500:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001 

        for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        for t in range(0,params['ntrain'][1]):
            for b in range(params['nb_batch']):
                # Forward pass: Compute predicted y by passing x to the model
                inp_concat = torch.cat((x[b,:,:], modelRINN.Dyn_net.y_aug[b,:,:]), dim=1)
                pred, grad, inp, aug_inp = modelRINN(inp_concat,dt)
                # Compute and print loss
                loss1 = criterion(grad[:,:1], z[b,:,:])
                loss2 = criterion(pred[:-1,:1] , aug_inp[1:,:1])
                loss3 = criterion(pred[:-1,1:] , aug_inp[1:,1:])
                loss =  0.0*loss1+1.0*loss2 + 1.0*loss3
                if t%1000==0:
                    print('Training L63 NbedDyn model', t,loss)
                torch.save(modelRINN.state_dict(), params['path'] + params['file_name']+'.pt')
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        torch.save(modelRINN.state_dict(), params['path'] + params['file_name'])
        return model, modelRINN, aug_inp_data
def train_NbedDyn_model_SLA(params,model,modelRINN,X_train,Grad_t):
        dt = params['dt_integration']
        aug_inp_data = []
        x = torch.from_numpy(X_train).float()
        z = torch.from_numpy(Grad_t).float()
        
        
        criterion = torch.nn.MSELoss(reduction='elementwise_mean')
        
        if params['pretrained'] :
            modelRINN.load_state_dict(torch.load(path + file_name+'.pt'))
        optimizer = torch.optim.Adam(model.parameters())
        
        for param_group in optimizer.param_groups:
                print(param_group['lr'])
        for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001      
        
        for t in range(params['ntrain'][0]):
            for b in range(params['nb_batch']):
                optimizer.zero_grad()      
                inp_concat = torch.cat((x[b,:,:], modelRINN.Dyn_net.y_aug[b,:,:]), dim=1)
                pred1, grad1, inp, aug_inp = modelRINN(inp_concat,dt)
                if params['get_latent_train']:
                    aug_inp_data.append(aug_inp.detach())
                loss1 = criterion(grad1[:,:params['dim_input']], z[b,:,:])
                loss2 = criterion(pred1[:-1,:], inp_concat[1:,:])
                loss = 0.9*loss1+0.1*loss2
                if t%1000==0:
                        print('Training SLA NbedDyn model', t,loss)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        torch.save(modelRINN.state_dict(), params['path'] + params['file_name']+'.pt')
        return model, modelRINN, aug_inp_data
def train_NbedDyn_model_Linear(params,model,modelRINN,X_train,Grad_t):    
        dt = params['dt_integration']
        aug_inp_data = []
        x = torch.from_numpy(X_train).float()
        z = torch.from_numpy(Grad_t).float()
        
        
        criterion = torch.nn.MSELoss(reduction='elementwise_mean')
        
        if params['pretrained'] :
            modelRINN.load_state_dict(torch.load(path + file_name+'.pt'))
        optimizer = torch.optim.Adam(model.parameters())
        
        for param_group in optimizer.param_groups:
                print(param_group['lr'])
        for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1      
        
        for t in range(params['ntrain'][0]):
            # Forward pass: Compute predicted y by passing x to the model
            for b in range(params['nb_batch']):
                aug_inp = torch.cat((x[b,:,:],modelRINN.Dyn_net.y_aug[b,:,:]),dim = -1)
                pred, grad, inp, aug_inp = modelRINN(aug_inp,dt)
                if params['get_latent_train']:
                    aug_inp_data.append(aug_inp.detach())
                # Compute and print loss
                loss1 = criterion(grad[:,:1], z[b,:,:])
                loss2 = criterion(pred[:-1,:], aug_inp[1:,:])
                loss = 1.0*loss1+1.0*loss2
                if t%1000==0:
                       print('Training Linear NbedDyn model', t,loss)
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        torch.save(modelRINN.state_dict(), params['path'] + params['file_name']+'.pt')
        return model, modelRINN, aug_inp_data
    