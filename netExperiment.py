import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt

from net import Net

class NetExperiment:
    num_exps = 0

    def __init__(self, n, num_trials=100, resize=False, name=None, description=""):
        empty_dict = {'start_nodes':[], 'weights': [], 'class': []}
        self.data = pd.DataFrame(empty_dict)
        NetExperiment.num_exps += 1
        if name == None:
            self.name = 'experiment' + str(NetExperiment.num_exps)
        else:
            self.name = name
        self.description = description
        self.resize=False
        self.num_trials = num_trials
        self.resize = resize
        self.n = n

    def to_net(self, index):
        '''
        Takes datum from dataframe at certain index and returns a Net obj with those nodes and weights
        '''
        def clean_string(s):
            return s.replace('\n', '').replace('[', '').replace(']', '')

        net = Net(self.n)
        net.nodes = np.fromstring(self.data['start_nodes'][index][1:-1], sep=' ')
        net.weights = np.fromstring(clean_string(self.data['weights'][index]), sep=' ').reshape((self.n, self.n))
        return net
        

    def run(self):
        '''
        Classifies num_trials amount of nets of size n with no input/output layers. Stores the data in self.data
        '''
        start_nodes_list = []
        weights_list = []
        classification_list = []
        for i in range(self.num_trials):
            net = Net(self.n)
            start_nodes_list.append(net.nodes)
            weights_list.append(net.weights)
            classification_list.append(NetExperiment.classify(net, resize=self.resize))
        self.data = pd.DataFrame({ 'class': classification_list, 'start_nodes': start_nodes_list, 'weights': weights_list})

    def write_csv(self):
        '''
        Writes data to csv file
        Also writes other class attributes to data frame
        puts file in experiments folder
        '''
        path = NetExperiment.path_from_name(self.name)
        metadata=pd.DataFrame({'name': self.name, 'description' : self.description, 'n' : self.n, 'trials': self.num_trials, 'resize': self.resize}, index=[0])
        metadata.to_csv(path, index=False)
        self.data.to_csv(path, mode='a')

        
    @staticmethod
    def read_csv(name):
        '''
        Reads data from csv file and returns new NetExperiment with dataframe and metadata
        reads from experiments folder
        '''
        path = NetExperiment.path_from_name(name)

        metadata_dtype_dict = {
            'n': int,
            'name': str,
            'description': str,
            'trials': int,
            'resize': bool
        }
        df = pd.read_csv(path, dtype=metadata_dtype_dict, nrows=1)

        exp = NetExperiment(df['n'][0], name=df['name'][0], description=df['description'][0], num_trials=df['trials'], resize=df['resize'][0])

        dtype_dict = {
            'class': int,
            'start_nodes': object,
            'weights': object
        }
        exp.data = pd.read_csv(path, dtype=dtype_dict, skiprows=2).drop('Unnamed: 0', axis=1)

        return exp

    @staticmethod
    def classify(net: Net, epsilon=.01, max_iters=1000, max_cycles=200, stim=[], resize=False):
        '''
        Returns the classification of a net's convergence
        1 if stable state
        for a cycle of length k, returns k where k is in [2,inf]
        0 if classification unknown
        -1 if all weights 1
        -2 if all weights 0
        '''

        cycle_dection_nodes = None
        i = 0
        while i < max_iters:
            last_nodes = net.nodes
            if resize:
                net.step_resize(stim)
            else:    
                net.step(stim)
            if abs(net.nodes[0]) < epsilon:
                if np.all(abs(net.nodes) < epsilon):
                    return -2
            if abs(net.nodes[0] - 1 < epsilon):
                if np.all(np.abs(net.nodes - 1) < epsilon):
                    return -1
            if abs(net.nodes[0] - last_nodes[0]) < epsilon:
                if np.all(np.abs(net.nodes - last_nodes) < epsilon):
                    return 1
            if cycle_dection_nodes is not None:
                if abs(cycle_dection_nodes[0] - net.nodes[0]) < epsilon:
                    if np.all(np.abs(cycle_dection_nodes - net.nodes) < epsilon):
                        return i % max_cycles
            if i % max_cycles == 0:
                cycle_dection_nodes = net.nodes
            i+=1
        return 0

    @staticmethod
    def path_from_name(name):
        path = 'experiments/' + name
        if not path.endswith('.csv'):
            path += '.csv'
        return path

    def hist(self, ax=None):
        if ax == None:
            ax = plt.gca()

        bins = np.array(range(10))
        ax.set_ylabel('n = ' + str(self.n))
        ax.set_xticks(bins)
        ax.set_ylim(0, 1)
        # dat = self.data['class'][self.data['class'] > 1]
        dat = self.data['class']
        n, _ , _ = ax.hist(dat, bins - .5, ec="red", density=True)
        for i in range(len(n)):
            ax.text(bins[i], n[i], str(round(n[i], 2)), ha='center', va='bottom')


