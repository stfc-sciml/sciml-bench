import sys
import torch
import ase
import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from torch.nn import functional as F
from torch_geometric.data import Data

sys.path.insert(0, '/people/herm379/exalearn/IPU_trained_models/pnnl_sandbox/schnet')
from model import SchNet


def schnet_eg(atoms, net, device='cpu'):
    """
    Takes in ASE atoms and loaded net and predicts energy and gradients
    args: atoms (ASE atoms object), net (loaded trained Schnet model)
    return: predicted energy (eV), predicted gradients (eV/angstrom)
    """
    types = {'H': 0, 'O': 1}
    atom_types = [1, 8]
    
    #center = False
    #if center:
    #    pos = atoms.get_positions() - atoms.get_center_of_mass()
    #else:
    
    pos = atoms.get_positions()
    pos = torch.tensor(pos, dtype=torch.float) #coordinates
    size = int(pos.size(dim=0)/3)
    type_idx = [types.get(i) for i in atoms.get_chemical_symbols()]
    atomic_number = atoms.get_atomic_numbers()
    z = torch.tensor(atomic_number, dtype=torch.long)
    x = F.one_hot(torch.tensor(type_idx, dtype=torch.long),
                              num_classes=len(atom_types))
    data = Data(x=x, z=z, pos=pos, size=size, batch=torch.tensor(np.zeros(size*3), dtype=torch.int64), idx=1)

    data = data.to(device)
    data.pos.requires_grad = True
    e = net(data)
    f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=True)[0].cpu().data.numpy()
    e = e.cpu().data.numpy()

    return e.item()/23.06035, f/23.06035

class SchnetCalculator(Calculator):
    """ASE interface to trained model
    """
    implemented_properties = ['forces', 'energy']
    nolabel = True

    def __init__(self, best_model, atoms=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        
        state=torch.load(best_model)
    
        num_gaussians = state['basis_expansion.offset'].shape[0]
        num_filters = state['interactions.0.mlp.0.weight'].shape[0]
        num_interactions = len([key for key in state.keys() if '.lin.bias' in key])

        net = SchNet(num_features = num_filters,
                     num_interactions = num_interactions,
                     num_gaussians = num_gaussians,
                     cutoff = 6.0)

        net.load_state_dict(state)

        self.net = net
        self.atoms = atoms

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties
        
        if atoms is not None:
            self.atoms = atoms.copy()
        
        Calculator.calculate(self, atoms, properties, system_changes)

        energy, gradients = schnet_eg(self.atoms, self.net)
        self.results['energy'] = energy
        self.results['forces'] = -gradients
        
