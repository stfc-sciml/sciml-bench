import torch
import numpy as np
import pandas as pd

torch.pi = torch.acos(torch.zeros(1)).item() * 2 

def force_magnitude_error(actual, pred):
    # ||f_hat|| - ||f||
    return torch.sub(torch.norm(pred, dim=1), torch.norm(actual, dim=1))

def force_angular_error(actual, pred):
    # cos^-1( f_hat/||f_hat|| • f/||f|| ) / pi
    # batched dot product obtained with torch.bmm(A.view(-1, 1, 3), B.view(-1, 3, 1))
    
    a = torch.norm(actual, dim=1)
    p = torch.norm(pred, dim=1)
    
    return torch.div(torch.acos(torch.bmm(torch.div(actual.T, a).T.view(-1, 1, 3), torch.div(pred.T, p).T.view(-1, 3,1 )).view(-1)), torch.pi)

def get_max_forces(data, forces):
    # data: data from DataLoader
    # forces: data.f for actual, f for pred
    start = 0
    f=[]
    for size in data.size.numpy()*3:
        f.append(np.abs(forces[start:start+size].numpy()).max())
        start += size
    return f


def infer(loader, net, forces=True, energies=True, force_type='max', device='cpu'):
    # force_type (str): ['max', 'all', 'error']
    f_actual = []
    e_actual = []
    e_pred = []
    f_pred = []
    size = []

    for data in loader:
        data = data.to(device)
        #size += data.size.tolist()
        size += data['size'].tolist()
        if energies:
            e_actual += data.y.tolist()
            e = net(data)
            e_pred += e.tolist()

        # get predicted values
        if forces:
            data.pos.requires_grad = True
            f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]
        
            if force_type == 'max':
                f_actual += get_max_forces(data, data.f)
                f_pred += get_max_forces(data, f)
            elif force_type == 'all':
                f_actual += data.f.squeeze().tolist()
                f_pred += f.squeeze().tolist()
            elif force_type == 'error':
                f_actual += force_magnitude_error(data.f, f)
                f_pred += force_angular_error(data.f, f)
            else:
                raise ValueError('Not implemented')
                
    # return as dataframe
    if energies == True and forces == False:
 
        # Relative difference between e_actual and e_pred
        num_e_actual = np.array(e_actual)
        num_e_pred = np.array(e_pred)
        num_error = abs((num_e_actual - num_e_pred)/num_e_actual)
        error = np.ndarray.tolist(num_error)
        
        return pd.DataFrame({'size': size, 'e_actual': e_actual, 'e_pred': e_pred, 'error': error})
    
    if energies == True and forces == True:
        if force_type=='error':
            return pd.DataFrame({'size': size, 'e_actual': e_actual, 'e_pred': e_pred}), pd.DataFrame({'f_mag_error': f_actual, 'f_ang_error': f_pred})
        else:
            return pd.DataFrame({'size': size, 'e_actual': e_actual, 'e_pred': e_pred}), pd.DataFrame({'f_actual': f_actual, 'f_pred': f_pred})

        
def infer_with_error(loader, net):
    fme_mse, fme_mae = [],[]
    fae_mse, fae_mae = [],[]
    e_actual = []
    e_pred = []
    f_pred = []
    size = []
    for data in loader:
        # extract ground truth values
        e_actual += data.y.tolist()
        size += data['size'].tolist()

        # get predicted values
        data.pos.requires_grad = True
        e = net(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]

        # reshape f to ragged tensor
        # compute f errors for each sample
        start = 0
        for dsize in data['size'].numpy()*3:
            f_ragged = f[start:start+dsize]
            f_act = data.f[start:start+dsize]

            fme = force_magnitude_error(f_act, f_ragged)
            fae = force_angular_error(f_act, f_ragged)

            fme_mae += [torch.mean(torch.abs(fme)).tolist()]
            fme_mse += [torch.mean(torch.square(fme)).tolist()]
            fae_mae += [torch.mean(torch.abs(fae)).tolist()]
            fae_mse += [torch.mean(torch.square(fae)).tolist()]

            start += dsize

        # get properties
        e_pred += e.tolist()


    return pd.DataFrame({'size': size,
                         'e_actual': e_actual, 'e_pred': e_pred,
                         'fme_mae': fme_mae, 'fae_mae': fae_mae,
                         'fme_mse': fme_mse, 'fae_mse': fae_mse})
