from matplotlib import pyplot as plt # to be deleted
import pyperclip

import numpy as np
from abc import ABC, abstractmethod

from net import Net

class Environment(ABC):
    def __init__(self, input_len, output_len):
        '''
        input_len: size of input to environmnet. This needs to be the same as OUTPUT of net
        type: int
        output_len: size of output to environmnet. This needs to be the same as INPUT of net
        type
        '''
        self.env_input_len = input_len
        self.env_output_len = output_len
        self.state = None
    
    @abstractmethod
    def step(self, net_output):
        '''
        Updates the environmnent based on the current state and the current output by the net
        Modifies the environments state variable
        '''
        pass
    
    @abstractmethod
    def get_net_input(self):
        '''
        Converts the environment state into values of the same shape as the net's input
        Returns a numpy array of length equal to net's input length
        First value of net input corresponds to the value returned by get_net_reward()
        '''
        pass

    @abstractmethod
    def get_net_reward(self):
        '''
        Returns the net reward level based on the current state
        '''
        pass

    def run(env, net : Net, iters=None):
        '''
        Itaeratively loops through environment steps and net steps
        '''
        if iters==None:
            plt.ion()
            while True:
                env.step(net.step(stim=env.get_net_input(), r=env.get_net_reward()))
                
                net.draw()
                net.draw_memory()
                # print(round(net.get_Nodes()[1], 3), env.get_net_reward())
                # print(net.get_Nodes(), 'e')
                pyperclip.copy('(' + str(round(net.get_Nodes()[1], 3)) + ',' + str(env.get_net_reward()) + '),')

                input()

        else:
            for i in range(iters):
                env.step(net.step(stim=env.get_net_input(), r=env.get_net_reward()))
            return env.get_net_reward()

    def run_threads(self, net : Net, env_speed=60, net_speed=60):
        '''
        Runs net and environemnt steps on separate threads
        Steps per second
        '''
        pass

    def save_env(self, name:str):
        '''
        Saves environment as pickle file in environemnts folder with name
        name: name of file
        type: str

        '''


class BasicEnv(Environment):
    def __init__(self):
        super().__init__(1, 0)
        self.state = {'age' : 0, 'r' : 0}
    
    def step(self, net_output):
        '''
        Steps the basic environement
        Increases 'age' attribute
        '''
        self.state['age'] += 1
        if net_output[0] > .5:
            self.state['r'] = 1
        else:
            self.state['r'] = -1

    def get_net_input(self):
        '''
        Converts the environment state into values of the same shape as the net's input
        Returns a numpy array of length equal to net's input length
        First value of net input corresponds to the value returned by get_net_reward()
        '''
        return []

    def get_net_reward(self):
        '''
        Returns the net reward level based on the current state
        '''
        return self.state['r']
