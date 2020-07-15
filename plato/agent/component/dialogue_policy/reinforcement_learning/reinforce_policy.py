"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Alexandros Papangelis"

from .. import dialogue_policy
from plato.agent.component.dialogue_policy.slot_filling_policy \
    import HandcraftedPolicy
from plato.domain.ontology import Ontology
from plato.domain.database import DataBase
from plato.dialogue.action import DialogueAct, DialogueActItem, Operator
from plato.dialogue.state import SlotFillingDialogueState
from plato.agent.component.user_simulator.\
    agenda_based_user_simulator.agenda_based_us import AgendaBasedUS
from copy import deepcopy

import numpy as np
import random
import os
import pickle

from .net import MLPNet
import torch 
from torch.distributions.categorical import Categorical
"""
ReinforcePolicy implements the REINFORCE algorithm for dialogue policy 
learning.
"""


class ReinforcePolicy(dialogue_policy.DialoguePolicy):

    def __init__(self, args):
        """
        Initialize parameters and internal structures

        :param args: the policy's arguments
        """

        super(ReinforcePolicy, self).__init__()

        self.ontology = None
        if 'ontology' in args:
            ontology = args['ontology']

            if isinstance(ontology, Ontology):
                self.ontology = ontology
            else:
                raise ValueError('ReinforcePolicy Unacceptable '
                                 'ontology type %s ' % ontology)
        else:
            raise ValueError('ReinforcePolicy: No ontology provided')

        self.database = None
        if 'database' in args:
            database = args['database']

            if isinstance(database, DataBase):
                self.database = database
            else:
                raise ValueError('ReinforcePolicy: Unacceptable '
                                 'database type %s ' % database)
        else:
            raise ValueError('ReinforcePolicy: No database provided')

        self.agent_id = args['agent_id'] if 'agent_id' in args else 0
        self.agent_role = \
            args['agent_role'] if 'agent_role' in args else 'system'

        domain = args['domain'] if 'domain' in args else None
        self.alpha = args['alpha'] if 'alpha' in args else 0.2
        self.gamma = args['gamma'] if 'gamma' in args else 0.95
        self.epsilon = args['epsilon'] if 'epsilon' in args else 0.95
        self.alpha_decay_rate = \
            args['alpha_decay'] if 'alpha_decay' in args else 0.995
        self.exploration_decay_rate = \
            args['epsilon_decay'] if 'epsilon_decay' in args else 0.9995

        self.IS_GREEDY = True

        self.policy_path = None

        self.weights = None
        self.sess = None

        # System and user expert policies (optional)
        self.warmup_policy = None
        self.warmup_simulator = None

        if self.agent_role == 'system':
            # Put your system expert policy here
            self.warmup_policy = HandcraftedPolicy({
                    'ontology': self.ontology})

        elif self.agent_role == 'user':
            usim_args = \
                dict(
                    zip(['ontology', 'database'],
                        [self.ontology, self.database]))
            # Put your user expert policy here
            self.warmup_simulator = AgendaBasedUS(usim_args)

        self.tf_scope = "policy_" + self.agent_role + '_' + str(self.agent_id)

        # Default value
        self.is_training = True

        # Extract lists of slots that are frequently used
        self.informable_slots = \
            deepcopy(list(self.ontology.ontology['informable'].keys()))
        self.requestable_slots = \
            deepcopy(self.ontology.ontology['requestable'])
        self.system_requestable_slots = \
            deepcopy(self.ontology.ontology['system_requestable'])

        if not domain:
            # Default to CamRest dimensions
            self.NStateFeatures = 56

            # Default to CamRest actions
            self.dstc2_acts = ['inform', 'offer', 'request', 'canthelp',
                               'affirm', 'negate', 'deny', 'ack', 'thankyou',
                               'bye', 'reqmore', 'hello', 'welcomemsg',
                               'expl-conf', 'select', 'repeat', 'reqalts',
                               'confirm-domain', 'confirm']
        else:
            # Try to identify number of state features
            if domain in ['CamRest', 'SFH', 'SlotFilling']:
                d_state = \
                    SlotFillingDialogueState(
                        {'slots': self.system_requestable_slots})

                # Plato does not use action masks (rules to define which
                # actions are valid from each state) and so training can
                # be harder. This becomes easier if we have a smaller
                # action set.

                # Sub-case for CamRest
                if domain == 'CamRest':
                    # Does not include inform and request that are modelled
                    # together with their arguments
                    self.dstc2_acts_sys = ['offer', 'canthelp', 'affirm',
                                           'deny', 'ack', 'bye',
                                           'reqmore', 'welcomemsg',
                                           'expl-conf', 'select', 'repeat',
                                           'confirm-domain', 'confirm']

                    # Does not include inform and request that are modelled
                    # together with their arguments
                    self.dstc2_acts_usr = ['affirm', 'negate', 'deny', 'ack',
                                           'thankyou', 'bye',
                                           'reqmore', 'hello', 'expl-conf',
                                           'repeat', 'reqalts', 'restart',
                                           'confirm']

            else:
                print('Warning! domain has not been defined. Using '
                      'Slot-Filling dialogue State')
                d_state = \
                    SlotFillingDialogueState({'slots': self.informable_slots})

            d_state.initialize()
            self.NStateFeatures = len(self.encode_state(d_state))

            print('Reinforce policy {0} automatically determined '
                  'number of state features: {1}'
                  .format(self.agent_role, self.NStateFeatures))

        if domain == 'CamRest' and self.dstc2_acts_sys:
            if self.agent_role == 'system':
                self.NActions = \
                    len(self.dstc2_acts_sys) + \
                    len(self.requestable_slots) + \
                    len(self.system_requestable_slots)

                self.NOtherActions = \
                    len(self.dstc2_acts_usr) + \
                    2 * len(self.requestable_slots)

            elif self.agent_role == 'user':
                self.NActions = \
                    len(self.dstc2_acts_usr) + \
                    2 * len(self.requestable_slots)

                self.NOtherActions = \
                    len(self.dstc2_acts_sys) + \
                    len(self.requestable_slots) + \
                    len(self.system_requestable_slots)

        else:
            if self.agent_role == 'system':
                self.NActions = \
                    3 + len(self.system_requestable_slots) + \
                    len(self.requestable_slots)

                self.NOtherActions = \
                    2 + len(self.requestable_slots) +\
                    len(self.requestable_slots)

            elif self.agent_role == 'user':
                self.NActions = \
                    2 + len(self.requestable_slots) + \
                    len(self.requestable_slots)

                self.NOtherActions = \
                    3 + len(self.system_requestable_slots) + \
                    len(self.requestable_slots)

        print('Reinforce {0} policy Number of Actions: {1}'
              .format(self.agent_role, self.NActions))

    def initialize(self, args):
        """
        Initialize internal structures at the beginning of each dialogue

        :return: Nothing
        """

        if 'is_training' in args:
            self.is_training = bool(args['is_training'])

            if self.agent_role == 'user' and self.warmup_simulator:
                if 'goal' in args:
                    self.warmup_simulator.initialize({args['goal']})
                else:
                    print('WARNING ! No goal provided for Reinforce policy '
                          'user simulator @ initialize')
                    self.warmup_simulator.initialize({})

        if 'policy_path' in args:
            self.policy_path = args['policy_path']

        if 'learning_rate' in args:
            self.alpha = args['learning_rate']
            
        if 'learning_decay_rate' in args:
            self.alpha_decay_rate = args['learning_decay_rate']

        if 'discount_factor' in args:
            self.gamma = args['discount_factor']

        if 'exploration_rate' in args:
            self.alpha = args['exploration_rate']

        if 'exploration_decay_rate' in args:
            self.exploration_decay_rate = args['exploration_decay_rate']

        # if self.weights is None:
        #     self.weights = np.random.rand(self.NStateFeatures, self.NActions)
        self.policy_loss_coef = 1.0
        self.net = MLPNet(self.NStateFeatures, self.NActions)
        # init net optimizer and its lr scheduler
        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.02)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1000, gamma=0.9)

    def restart(self, args):
        """
        Re-initialize relevant parameters / variables at the beginning of each
        dialogue.

        :return: nothing
        """

        if self.agent_role == 'user' and self.warmup_simulator:
            if 'goal' in args:
                self.warmup_simulator.initialize(args)
            else:
                print('WARNING! No goal provided for Reinforce '
                      'policy user simulator @ restart')
                self.warmup_simulator.initialize({})

    def next_action(self, state):
        """
        Consults the policy to produce the agent's response

        :param state: the current dialogue state
        :return: a list of dialogue acts, representing the agent's response
        """

        if self.is_training and random.random() < self.epsilon:
            if random.random() < 0.5:
                print('--- {0}: Selecting warmup action.'
                      .format(self.agent_role))

                if self.agent_role == 'system':
                    return self.warmup_policy.next_action(state)

                else:
                    self.warmup_simulator.receive_input(
                        state.user_acts, state.user_goal)
                    return self.warmup_simulator.respond()

            else:
                print('--- {0}: Selecting random action.'
                      .format(self.agent_role))
                return self.decode_action(
                    random.choice(
                        range(0, self.NActions)),
                    self.agent_role == "system")

        # Probabilistic policy: Sample from action wrt probabilities
        with torch.no_grad():
            probs_t = self.calculate_policy((self.encode_state(state)))
        probs = probs_t.numpy()[0]
        # probs = self.calculate_policy(self.encode_state(state))

        if any(np.isnan(probs)):
            print('WARNING! NAN detected in action probabilities! Selecting '
                  'random action.')
            return self.decode_action(
                random.choice(range(0, self.NActions)),
                self.agent_role == "system")

        if self.IS_GREEDY:
            # Get greedy action
            max_pi = max(probs)
            maxima = [i for i, j in enumerate(probs) if j == max_pi]

            # Break ties randomly
            if maxima:
                sys_acts = \
                    self.decode_action(
                        random.choice(maxima), self.agent_role == 'system')
            else:
                print(
                    f'--- {self.agent_role}: Warning! No maximum value '
                    f'identified for policy. Selecting random action.')
                return self.decode_action(
                    random.choice(
                        range(0, self.NActions)),
                    self.agent_role == 'system')
        else:
            # Pick from top 3 actions
            top_3 = np.argsort(-probs)[0:2]
            sys_acts = \
                self.decode_action(
                    random.choices(
                        top_3, probs[top_3])[0], self.agent_role == 'system')

        return sys_acts

    def calculate_policy(self, state):
        """
        Calculates the probabilities for each action from the given state

        :param state: the current dialogue state
        :return: probabilities of actions
        """
        t_state = torch.tensor(state).view(1,-1).float()
        pdparam = self.net(t_state)
        return pdparam

    def train(self, memory):
        """
        Train the policy network

        :param dialogues: dialogue experience
        :return: nothing
        """
        # If called by accident
        print("update the policy dialogue")
        if not self.is_training:
            return
        # for dialogue in dialogues:
        #     state_enc = [self.encode_state(turn['state']) for turn in dialogue]

        #     if len(state_enc) != self.NStateFeatures:
        #         raise ValueError(f'Reinforce dialogue policy '
        #                             f'{self.agent_role} mismatch in state'
        #                             f'dimensions: State Features: '
        #                             f'{self.NStateFeatures} != State '
        #                             f'Encoding Length: {len(state_enc)}')

        #     # Calculate the gradients

        #     # Call policy again to retrieve the probability of the
        #     # action taken
        #     pdparams = self.calculate_policy(state_enc)
        #     advs = self.calc_ret_advs(dialogue)
        #     loss = self.calc_policy_loss(dialogue, pdparams, advs)
        #     self.net.train_step(loss, self.optim)
        #         # discount *= self.gamma

        # if self.alpha > 0.01:
        #     self.alpha *= self.alpha_decay_rate

        # if self.epsilon > 0.5:
        #     self.epsilon *= self.exploration_decay_rate
        # print(f'REINFORCE train, alpha: {self.alpha}, epsilon: {self.epsilon}')
        batch = self.sample(memory)
        pdparams = self.calc_pdparam_batch(batch)
        advs = self.calc_ret_advs(batch)
        loss = self.calc_policy_loss(batch, pdparams, advs)
        self.net.train_step(loss, self.optim)


    def calc_ret_advs(self, batch):
        '''Calculate plain returns; which is generalized to advantage in ActorCritic'''
        rets = self.calc_returns(batch['rewards'], batch['dones'], self.gamma)
        advs = rets
        return advs

    # def calc_policy_loss(self, dialogue, pdparams, advs):
    #     '''Calculate the actor's policy loss'''
    #     action_pd = Categorical(pdparams)
    #     actions = torch.tensor([self.encode_action(turn['action'],self.agent_role == 'system') for turn in dialogue])
    #     log_probs = action_pd.log_prob(actions)
    #     policy_loss = - self.policy_loss_coef * (log_probs * advs).mean()
    #     return policy_loss 
    
    def calc_policy_loss(self, batch, pdparams, advs):
        '''Calculate the actor's policy loss'''
        action_pd = Categorical(pdparams)
        actions = batch['actions']
        log_probs = action_pd.log_prob(actions)
        policy_loss = - self.policy_loss_coef * (log_probs * advs).mean()
        # if self.entropy_coef_spec:
        #     entropy = action_pd.entropy().mean()
        #     self.body.mean_entropy = entropy  # update logging variable
        #     policy_loss += (-self.body.entropy_coef * entropy)
        # logger.debug(f'Actor policy loss: {policy_loss:g}')
        return policy_loss

    def calc_pdparam_batch(self, batch):
        '''Efficiently forward to get pdparam and by batch for loss computation'''
        states = batch['states']
        pdparam = self.calc_pdparam(states)
        return pdparam

    def calc_pdparam(self, x, net=None):
        '''
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        net = self.net if net is None else net
        pdparam = net(x)
        return pdparam

    def sample(self,memory):
        '''Samples a batch from memory'''
        batch = memory.sample()
        batch = self.to_torch_batch(batch)
        return batch

    def to_torch_batch(self, batch):
        '''Mutate a batch (dict) to make its values from numpy into PyTorch tensor'''
        for k in batch:
            batch[k] = np.concatenate(batch[k])
            batch[k] = torch.from_numpy(batch[k].astype(np.float32))
        return batch

    def calc_returns(self, rewards, dones, gamma):
        '''
        Calculate the simple returns (full rollout) i.e. sum discounted rewards up till termination
        '''
        T = len(rewards)
        rets = torch.zeros_like(rewards)
        future_ret = torch.tensor(0.0, dtype=rewards.dtype)
        not_dones = 1 - dones
        for t in reversed(range(T)):
            rets[t] = future_ret = rewards[t] + gamma * future_ret * not_dones[t]
        return rets

    def encode_state(self, state):
        """
        Encodes the dialogue state into a vector.

        :param state: the state to encode
        :return: int - a unique state encoding
        """

        temp = [int(state.is_terminal_state), int(state.system_made_offer)]

        if self.agent_role == 'user':
            # The user agent needs to know which constraints and requests
            # need to be communicated and which of them
            # actually have.
            if state.user_goal:
                for c in self.informable_slots:
                    if c != 'name':
                        if c in state.user_goal.constraints:
                            temp.append(1)
                        else:
                            temp.append(0)

                for c in self.informable_slots:
                    if c != 'name':
                        if c in state.user_goal.actual_constraints and \
                                state.user_goal.actual_constraints[c].value:
                            temp.append(1)
                        else:
                            temp.append(0)

                for r in self.requestable_slots:
                    if r in state.user_goal.requests:
                        temp.append(1)
                    else:
                        temp.append(0)

                for r in self.requestable_slots:

                    if r in state.user_goal.actual_requests and \
                            state.user_goal.actual_requests[r].value:
                        temp.append(1)
                    else:
                        temp.append(0)

            else:
                temp += [0] * 2 * (len(self.informable_slots) - 1 +
                                   len(self.requestable_slots))

        if self.agent_role == 'system':
            for value in state.slots_filled.values():
                # This contains the requested slot
                temp.append(1) if value else temp.append(0)

            for r in self.requestable_slots:
                temp.append(1) if r == state.requested_slot else temp.append(0)

        return temp

    def encode_action(self, actions, system=True):
        """
        Encode the action, given the role. Note that does not have to match
        the agent's role, as the agent may be encoding another agent's action
        (e.g. a system encoding the previous user act).

        :param actions: actions to be encoded
        :param system: whether the role whose action we are encoding is a
                       'system'
        :return: the encoded action
        """

        # TODO: Handle multiple actions
        if not actions:
            print('WARNING: Reinforce dialogue policy action encoding called '
                  'with empty actions list (returning 0).')
            actions = self.decode_action(random.choice(range(0, self.NActions)),self.agent_role == "system")

        action = actions[0]
        if system:
            if self.dstc2_acts_sys and action.intent in self.dstc2_acts_sys:
                return self.dstc2_acts_sys.index(action.intent)

            if action.intent == 'request':
                return len(self.dstc2_acts_sys) + \
                       self.system_requestable_slots.index(
                           action.params[0].slot)

            if action.intent == 'inform':
                return len(self.dstc2_acts_sys) + \
                       len(self.system_requestable_slots) + \
                       self.requestable_slots.index(action.params[0].slot)
        else:
            if self.dstc2_acts_usr and action.intent in self.dstc2_acts_usr:
                return self.dstc2_acts_usr.index(action.intent)

            if action.intent == 'request':
                return len(self.dstc2_acts_usr) + \
                       self.requestable_slots.index(action.params[0].slot)

            if action.intent == 'inform':
                return len(self.dstc2_acts_usr) + \
                       len(self.requestable_slots) + \
                       self.requestable_slots.index(action.params[0].slot)

        # Default fall-back action
        print('Reinforce ({0}) olicy action encoder warning: Selecting '
              'default action (unable to encode: {1})!'
              .format(self.agent_role, action))
        return -1

    def decode_action(self, action_enc, system=True):
        """
        Decode the action, given the role. Note that does not have to match
        the agent's role, as the agent may be decoding another agent's action
        (e.g. a system decoding the previous user act).

        :param action_enc: action encoding to be decoded
        :param system: whether the role whose action we are decoding is a
                       'system'
        :return: the decoded action
        """

        if system:
            if action_enc < len(self.dstc2_acts_sys):
                return [DialogueAct(self.dstc2_acts_sys[action_enc], [])]

            if action_enc < len(self.dstc2_acts_sys) + \
                    len(self.system_requestable_slots):
                return [DialogueAct(
                    'request',
                    [DialogueActItem(
                        self.system_requestable_slots[
                            action_enc - len(self.dstc2_acts_sys)],
                        Operator.EQ, '')])]

            if action_enc < len(self.dstc2_acts_sys) + \
                    len(self.system_requestable_slots) + \
                    len(self.requestable_slots):
                index = action_enc - len(self.dstc2_acts_sys) - \
                        len(self.system_requestable_slots)
                return [DialogueAct(
                    'inform',
                    [DialogueActItem(
                        self.requestable_slots[index], Operator.EQ, '')])]

        else:
            if action_enc < len(self.dstc2_acts_usr):
                return [DialogueAct(self.dstc2_acts_usr[action_enc], [])]

            if action_enc < len(self.dstc2_acts_usr) + \
                    len(self.requestable_slots):
                return [DialogueAct(
                    'request',
                    [DialogueActItem(
                        self.requestable_slots[
                            action_enc - len(self.dstc2_acts_usr)],
                        Operator.EQ,
                        '')])]

            if action_enc < len(self.dstc2_acts_usr) + \
                    2 * len(self.requestable_slots):
                return [DialogueAct(
                    'inform',
                    [DialogueActItem(
                        self.requestable_slots[
                            action_enc - len(self.dstc2_acts_usr) -
                            len(self.requestable_slots)],
                        Operator.EQ,
                        '')])]

        # Default fall-back action
        print('Reinforce dialogue policy ({0}) policy action decoder warning: '
              'Selecting default action (index: {1})!'
              .format(self.agent_role, action_enc))
        return [DialogueAct('bye', [])]

    def save(self, path=None):
        """
        Saves the policy model to the provided path

        :param path: path to save the model to
        :return:
        """

        # Don't save if not training
        if not self.is_training:
            return

        if not path:
            path = 'models/policies/reinforce.pkl'
            print('No policy file name provided. Using default: {0}'
                  .format(path))

        # If the directory does not exist, create it
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        obj = {'weights': self.weights,
               'alpha': self.alpha,
               'alpha_decay_rate': self.alpha_decay_rate,
               'epsilon': self.epsilon,
               'exploration_decay_rate': self.exploration_decay_rate}

        with open(path, 'wb') as file:
            pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

    def load(self, path=None):
        """
        Load the policy model from the provided path

        :param path: path to load the model from
        :return:
        """

        if not path:
            print('No dialogue policy loaded.')
            return

        if isinstance(path, str):
            if os.path.isfile(path):
                with open(path, 'rb') as file:
                    obj = pickle.load(file)

                    if 'weights' in obj:
                        self.weights = obj['weights']

                    if 'alpha' in obj:
                        self.alpha = obj['alpha']

                    if 'alpha_decay_rate' in obj:
                        self.alpha_decay_rate = obj['alpha_decay_rate']

                    if 'epsilon' in obj:
                        self.epsilon = obj['epsilon']

                    if 'exploration_decay_rate' in obj:
                        self.exploration_decay_rate = \
                            obj['exploration_decay_rate']

                    print('Reinforce policy loaded from {0}.'
                          .format(path))

            else:
                print('Warning! Reinforce policy file %s not found'
                      % path)
        else:
            print('Warning! Unacceptable value for Reinforce policy '
                  'file name: %s ' % path)
