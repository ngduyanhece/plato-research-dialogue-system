from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pydash as ps


def get_nn_name(uncased_name):
    '''Helper to get the proper name in PyTorch nn given a case-insensitive name'''
    for nn_name in nn.__dict__:
        if uncased_name.lower() == nn_name.lower():
            return nn_name
    raise ValueError(f'Name {uncased_name} not found in {nn.__dict__}')


class Net(ABC):
    '''Abstract Algorithm class to define the base methods'''

    def __init__(self,in_dim, out_dim):
        '''
        @param {int|list} in_dim is the input dimension(s) for the network. Usually use in_dim=state_dim
        @param {int|list} out_dim is the output dimension(s) for the network. Usually use out_dim=action_dim
        '''
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grad_norms = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self):
        '''The forward step for a specific network architecture'''
        raise NotImplementedError

    def train_step(self, loss, optim):
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss
    
    def store_grad_norms(self):
        raise NotImplementedError


class MLPNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network
    If more than 1 output tensors, will create a self.model_tails instead of making last layer part of self.model
    '''
    
    def __init__(self, in_dim, out_dim):
        nn.Module.__init__(self)
        super().__init__(in_dim, out_dim)
        self.shared=True 
        self.hid_layers=[64]
        self.hid_layers_activation='selu'
        self.out_layer_activation=None
        self.init_fn=None
        self.clip_grad_val = 0.5
        self.loss_spec={'name': 'MSELoss'}
        self.update_type='replace'
        self.update_frequency=1,
        self.polyak_coef=0.0
        
        dims = [self.in_dim] + self.hid_layers
        self.model = self.build_fc_model(dims, self.hid_layers_activation)
        if ps.is_integer(self.out_dim):
            self.model_tail = self.build_fc_model([dims[-1], self.out_dim], self.out_layer_activation)
        else:
            if not ps.is_list(self.out_layer_activation):
                self.out_layer_activation = [self.out_layer_activation] * len(out_dim)
            assert len(self.out_layer_activation) == len(self.out_dim)
            tails = []
            for out_d, out_activ in zip(self.out_dim, self.out_layer_activation):
                tail = self.build_fc_model([dims[-1], out_d], out_activ)
                tails.append(tail)
            self.model_tails = nn.ModuleList(tails)
        LossClass = getattr(nn, get_nn_name(self.loss_spec['name']))
        loss_spec = ps.omit(self.loss_spec, 'name')
        self.loss_fn = LossClass(**loss_spec)
        self.to(self.device)
        self.train()
    
    def forward(self, x):
        '''The feedforward step'''
        x = self.model(x)
        return self.model_tail(x)
    
    def build_fc_model(self, dims, activation=None):
        '''Build a full-connected model by interleaving nn.Linear and activation_fn'''
        assert len(dims) >= 2, 'dims need to at least contain input, output'
        # shift dims and make pairs of (in, out) dims per layer
        dim_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for in_d, out_d in dim_pairs:
            layers.append(nn.Linear(in_d, out_d))
            if activation is not None:
                layers.append(self.get_activation_fn(activation))
        model = nn.Sequential(*layers)
        return model

    def get_activation_fn(self, activation):
        '''Helper to generate activation function layers for net'''
        ActivationClass = getattr(nn, get_nn_name(activation))
        return ActivationClass()

    def get_loss_fn(self, loss_spec):
        '''Helper to parse loss param and construct loss_fn for net'''
        LossClass = getattr(nn, get_nn_name(loss_spec['name']))
        loss_spec = ps.omit(loss_spec, 'name')
        loss_fn = LossClass(**loss_spec)
        return loss_fn
