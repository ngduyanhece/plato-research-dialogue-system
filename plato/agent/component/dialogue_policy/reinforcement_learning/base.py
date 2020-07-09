from abc import ABC, abstractmethod
import numpy as np


class Algorithm(ABC):
    '''Abstract Algorithm class to define the base methods'''

    def __init__(self, agent):
        '''
        @param {*} agent is the container for algorithm and related components, and interfaces with env.
        '''
        self.agent = agent
        self.init_algorithm_params()
    
     def calc_pdparam(self, x):
        '''
        To get the pdparam for action policy sampling, do a forward pass of the appropriate net, and pick the correct outputs.
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        raise NotImplementedError 
    
    def act(self, state):
        '''Standard act method.'''
        raise NotImplementedError

    def sample(self):
        '''Samples a batch from memory'''
        raise NotImplementedError

    def train(self):
        '''Implement algorithm train, or throw NotImplementedError'''
        raise NotImplementedError

    def update(self):
        '''Implement algorithm update, or throw NotImplementedError'''
        raise NotImplementedError

    def save(self, ckpt=None):
        '''Save net models for algorithm given the required property self.net_names'''
        raise NotImplementedError

    def load(self):
        raise NotImplementedError