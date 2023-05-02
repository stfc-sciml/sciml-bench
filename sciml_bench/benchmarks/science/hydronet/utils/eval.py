import os
import torch
import numpy as np


def get_error_distribution(err_list):
    """
    Compute the MAE and standard deviation of the errors in the examine set.
    """
    err_array = np.array(err_list)
    mae = np.average(np.abs(err_array))
    var = np.average(np.square(np.abs(err_array)-mae))
    return mae, np.sqrt(var)


def get_idx_to_add(net, examine_loader, mae, std, energy_coeff, 
                   split_file, al_step, device, min_nonmin,
                   max_to_add=0.15, error_tolerance=0.15,
                   savedir = './'):
    """
    Computes the normalized (by cluster size) errors for all entries in the examine set. It will add a max of
    max_to_add samples that are p < 0.15.
    """
    net.eval()
    all_errs = []
    for data in examine_loader:
        data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()

        e = net(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]
        energies_loss = torch.abs(data.y - e)
        f_red = torch.mean(torch.abs(data.f - f), dim=1)
        
        f_mean = torch.zeros_like(e)
        cluster_sizes = data.size
        for i in range(len(e)):            #loop over all clusters in batch
            energies_loss[i] /= cluster_sizes[i]
            f_mean[i] = torch.mean(torch.abs(torch.tensor(f_red[torch.sum(cluster_sizes[0:i]):torch.sum(cluster_sizes[0:i+1])])))
        
        total_err = (energy_coeff)*energies_loss + (1-energy_coeff)*f_mean
        total_err = total_err.tolist()
        all_errs += total_err
    
    with open(os.path.join(savedir, f'error_distribution_alstep{al_step}_{min_nonmin}.pkl'), 'wb') as f:
        pickle.dump(all_errs, f)    

    S = np.load(os.path.join(savedir, split_file))
    examine_idx = S["examine_idx"].tolist()
    
    cutoff = erfinv(1-error_tolerance) * std + mae
    n_samples_to_add = int(len(all_errs)*max_to_add)
    idx_highest_errors = np.argsort(np.array(all_errs))[-n_samples_to_add:]
    idx_to_add = [examine_idx[idx] for idx in idx_highest_errors if all_errs[idx]>=cutoff]
    
    return idx_to_add


def get_pred_loss(model, loader, energy_coeff, device, val=False):
    """
    Gets the total loss on the test/val datasets.
    If validation set, then return MAE and STD also
    """
    model.eval()
    total_ef_loss = []
    all_errs = []
    
    for data in loader:
        data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()

        e = model(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]

        ef_loss, e_loss, f_loss = energy_forces_loss(data, e, f, energy_coeff)
        with torch.no_grad():
            total_ef_loss.append(ef_loss.item())
        if val == True:
            energies_loss = torch.abs(data.y - e)
            f_red = torch.mean(torch.abs(data.f - f), dim=1)

            f_mean = torch.zeros_like(e)
            cluster_sizes = data['size'] #data.size
            for i in range(len(e)):            #loop over all clusters in batch
                energies_loss[i] /= cluster_sizes[i]
                f_mean[i] = torch.mean(torch.abs(torch.tensor(f_red[torch.sum(cluster_sizes[0:i]):torch.sum(cluster_sizes[0:i+1])])))

            total_err = (energy_coeff)*energies_loss + (1-energy_coeff)*f_mean
            total_err = total_err.tolist()
            all_errs += total_err
    
    ave_ef_loss = sum(total_ef_loss)/len(total_ef_loss)
    
    if val == False:
        return ave_ef_loss
    
    else:
        mae, stdvae = get_error_distribution(all_errs) #MAE and STD from EXAMINE SET
        return ave_ef_loss, mae, stdvae