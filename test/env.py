import math
import os
from copy import deepcopy

import time
from abc import ABC, abstractmethod

import numpy as np
import pydash as ps
from gym import spaces

from slm_lab.env.base import set_gym_space_attr
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from slm_lab.agent import Agent, Body
from slm_lab.experiment import analysis
from slm_lab.spec import spec_util

from plato.controller.basic_controller import BasicController
from plato.agent.component.user_simulator.agenda_based_user_simulator.\
    agenda_based_us import AgendaBasedUS
from plato.agent.conversational_agent.conversational_generic_agent import \
    ConversationalGenericAgent
from plato.agent.conversational_agent.conversational_single_agent import \
    ConversationalSingleAgent
from plato.controller import controller

def set_gym_space_attr(gym_space):
    '''Set missing gym space attributes for standardization'''
    if isinstance(gym_space, spaces.Box):
        setattr(gym_space, 'is_discrete', False)
    elif isinstance(gym_space, spaces.Discrete):
        setattr(gym_space, 'is_discrete', True)
        setattr(gym_space, 'low', 0)
        setattr(gym_space, 'high', gym_space.n)
    elif isinstance(gym_space, spaces.MultiBinary):
        setattr(gym_space, 'is_discrete', True)
        setattr(gym_space, 'low', np.full(gym_space.n, 0))
        setattr(gym_space, 'high', np.full(gym_space.n, 2))
    elif isinstance(gym_space, spaces.MultiDiscrete):
        setattr(gym_space, 'is_discrete', True)
        setattr(gym_space, 'low', np.zeros_like(gym_space.nvec))
        setattr(gym_space, 'high', np.array(gym_space.nvec))
    else:
        raise ValueError('gym_space not recognized')

class BaseEnv(ABC):
    '''
    The base Env class with API and helper methods. Use this to implement your env class that is compatible with the Lab APIs

    e.g. env_spec
    "env": [{
        "name": "PongNoFrameskip-v4",
        "frame_op": "concat",
        "frame_op_len": 4,
        "normalize_state": false,
        "reward_scale": "sign",
        "num_envs": 8,
        "max_t": null,
        "max_frame": 1e7
    }],
    '''

    def __init__(self, spec):
        self.env_spec = spec['env'][0]  # idx 0 for single-env
        # set default
        util.set_attr(self, dict(
            eval_frequency=10000,
            log_frequency=10000,
            frame_op=None,
            frame_op_len=None,
            image_downsize=(84, 84),
            normalize_state=False,
            reward_scale=None,
            num_envs=1,
        ))
        util.set_attr(self, spec['meta'], [
            'eval_frequency',
            'log_frequency',
        ])
        util.set_attr(self, self.env_spec, [
            'name',
            'frame_op',
            'frame_op_len',
            'image_downsize',
            'normalize_state',
            'reward_scale',
            'num_envs',
            'max_t',
            'max_frame',
        ])
        if util.get_lab_mode() == 'eval':  # override if env is for eval
            self.num_envs = ps.get(spec, 'meta.rigorous_eval')
        self.to_render = util.to_render()
        self._infer_frame_attr(spec)
        self._infer_venv_attr()
        self._set_clock()
        self.done = False
        self.total_reward = np.nan

    def _get_spaces(self, u_env):
        '''Helper to set the extra attributes to, and get, observation and action spaces'''
        observation_space = u_env.observation_space
        action_space = u_env.action_space
        set_gym_space_attr(observation_space)
        set_gym_space_attr(action_space)
        return observation_space, action_space

    def _get_observable_dim(self, observation_space):
        '''Get the observable dim for an agent in env'''
        state_dim = observation_space.shape
        if len(state_dim) == 1:
            state_dim = state_dim[0]
        return {'state': state_dim}

    def _get_action_dim(self, action_space):
        '''Get the action dim for an action_space for agent to use'''
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1
            action_dim = action_space.shape[0]
        elif isinstance(action_space, (spaces.Discrete, spaces.MultiBinary)):
            action_dim = action_space.n
        elif isinstance(action_space, spaces.MultiDiscrete):
            action_dim = action_space.nvec.tolist()
        else:
            raise ValueError('action_space not recognized')
        return action_dim

    def _infer_frame_attr(self, spec):
        '''Infer frame attributes'''
        seq_len = ps.get(spec, 'agent.0.net.seq_len')
        if seq_len is not None:  # infer if using RNN
            self.frame_op = 'stack'
            self.frame_op_len = seq_len
        if spec['meta']['distributed'] != False:  # divide max_frame for distributed
            self.max_frame = int(self.max_frame / spec['meta']['max_session'])

    def _infer_venv_attr(self):
        '''Infer vectorized env attributes'''
        self.is_venv = (self.num_envs is not None and self.num_envs > 1)

    def _is_discrete(self, action_space):
        '''Check if an action space is discrete'''
        return util.get_class_name(action_space) != 'Box'

    def _set_clock(self):
        self.clock_speed = 1 * (self.num_envs or 1)  # tick with a multiple of num_envs to properly count frames
        self.clock = Clock(self.max_frame, self.clock_speed)

    def _set_attr_from_u_env(self, u_env):
        '''Set the observation, action dimensions and action type from u_env'''
        self.observation_space, self.action_space = self._get_spaces(u_env)
        self.observable_dim = self._get_observable_dim(self.observation_space)
        self.action_dim = self._get_action_dim(self.action_space)
        self.is_discrete = self._is_discrete(self.action_space)

    def _update_total_reward(self, info):
        '''Extract total_reward from info (set in wrapper) into self.total_reward for single and vec env'''
        if isinstance(info, dict):
            self.total_reward = info['total_reward']
        else:  # vec env tuple of infos
            self.total_reward = np.array([i['total_reward'] for i in info])

    @abstractmethod
    @lab_api
    def reset(self):
        '''Reset method, return state'''
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def step(self, action):
        '''Step method, return state, reward, done, info'''
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def close(self):
        '''Method to close and cleanup env'''
        raise NotImplementedError

class Clock:
    '''Clock class for each env and space to keep track of relative time. Ticking and control loop is such that reset is at t=0 and epi=0'''

    def __init__(self, max_frame=int(1e7), clock_speed=1):
        self.max_frame = max_frame
        self.clock_speed = int(clock_speed)
        self.reset()

    def reset(self):
        self.t = 0
        self.frame = 0  # i.e. total_t
        self.epi = 0
        self.start_wall_t = time.time()
        self.batch_size = 1  # multiplier to accurately count opt steps
        self.opt_step = 0  # count the number of optimizer updates

    def get(self, unit='frame'):
        return getattr(self, unit)

    def get_elapsed_wall_t(self):
        '''Calculate the elapsed wall time (int seconds) since self.start_wall_t'''
        return int(time.time() - self.start_wall_t)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def tick(self, unit='t'):
        if unit == 't':  # timestep
            self.t += self.clock_speed
            self.frame += self.clock_speed
        elif unit == 'epi':  # episode, reset timestep
            self.epi += 1
            self.t = 0
        elif unit == 'opt_step':
            self.opt_step += self.batch_size
        else:
            raise KeyError


class CamrestEnvironment(BaseEnv):
    '''
    wrapper class for camrest env
    '''
    def __init__(self,spec):
        super(CamrestEnvironment, self).__init__(spec)
        self.action_dim = 0
        self.observation_dim = 0
        self.observable_dim = {'state': 12}
        self.is_discrete = True
        self.spec = spec
        self.observation_space = None 
        self.action_space = None
        self.ctrl = BasicController()
        self.config = '/Users/andy/Ftech/plato-research-dialogue-system/plato/example/config/application/CamRest_model_reinforce_policy_train.yaml'
        self.configuration = self.ctrl.arg_parse(['_', '-config', self.config])['cfg_parser']
        self.ca = ConversationalSingleAgent(self.configuration)
    
    @lab_api
    def reset(self):
        pass

    @lab_api
    def step(self):
        dact, dstate, dnext_state, reward, done =  self.ca.continue_dialogue()
        action = self.ca.dialogue_manager.policy.encode_action(dact)
        state = self.ca.dialogue_manager.policy.encode_state(dstate)
        next_state = self.ca.dialogue_manager.policy.encode_state(dnext_state)  
        return action, state, next_state, reward, done
    @lab_api
    def close(self):
        pass

class Session:
    '''A very simple Session that runs an RL loop'''
    def __init__(self, spec):
        self.spec = spec
        self.env = CamrestEnvironment(self.spec)
        body = Body(self.env, self.spec)
        self.agent = Agent(self.spec, body=body)
        logger.info(f'Initialized session')
    
    def run_rl(self):
        clock = self.env.clock
        state = self.env.reset()
        done = False
        while clock.get('frame') <= self.env.max_frame:
            if done:  # reset when episode is done
                clock.tick('epi')
                state = self.env.reset()
                done = False
            clock.tick('t')
            action, state, next_state, reward, done = self.env.step()
            self.agent.update(state, action, reward, next_state, done)
            if clock.get('frame') % self.env.log_frequency == 0:
                self.agent.body.ckpt(self.env, 'train')
                self.agent.body.log_summary('train')
    def close(self):
        self.agent.close()
        self.env.close()
        logger.info('Session done and closed.')

    def run(self):
        self.run_rl()
        # this will run SLM Lab's built-in analysis module and plot graphs
        self.data = analysis.analyze_session(self.spec, self.agent.body.train_df, 'train')
        self.close()
        return self.data

if __name__ == "__main__":
    spec_dict = util.read('/Users/andy/Ftech/plato-research-dialogue-system/plato/example/config/lab/reinforce.json')
    spec_name = 'reinforce_cartpole'
    spec = spec_dict[spec_name]
    spec['name'] = spec_name
    spec = spec_util.extend_meta_spec(spec, experiment_ts=None)
    os.environ['lab_mode'] = 'train'  # set to 'dev' for rendering
    # update the tracking indices
    spec_util.tick(spec, 'trial')
    spec_util.tick(spec, 'session')

    # initialize and run session
    session = Session(spec)
    session_metrics = session.run()