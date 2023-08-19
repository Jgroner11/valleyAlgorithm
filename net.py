import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from landscape import Memory
import my_networkx as my_nx # for updated drawing of labels using bezier curve calculations


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Net:
    def __init__(self, n, input_len = 0, output_len = 0):
        """
        Initialize a new Net object.

        :param n: Number of total nodes
        :input_len: Number of input nodes.
        :output_len: Number of output nodes.
        :raises ValueError: If there are less total nodes than input + output.
        """
        if input_len + output_len > n:
            raise ValueError('Not enough total nodes')
    
        self.n = n
        self.input_len = input_len
        self.output_len = output_len

        self.nodes = np.random.rand(n-self.input_len).T # Node 0 to node output_len - 1 are the output nodes
        self.weights = self.randomizeWeights()
        self.memory = Memory(n)

        
        self.last_stim = np.ones(self.input_len) * .5

        self.plt_data = {}
        
        

    def step(self, stim=[], r=0):
        """
        Compute and store value of nodes based on current stimulus, current nodes, and current weights
        :stim: external stimulus
        :type stim: python list or np array. Length must be equal to self.input_len
                    The first value of stim represents the reward the system is recieving ie net weights change in order to maximize stim[0]
        :r: reward that net is recieving
        :type r: float
        :return: the output nodes
        :rtype: np.array
        """

        self.memory.update(self.nodes, r)
        # self.randomWeightShift(magnitude=.25)
        self.randomizeWeights()   
        self.nodes = sigmoid(self.weights @ np.hstack((self.nodes, np.array(stim))))

        self.last_stim = np.array(stim)
        return self.nodes[:self.output_len]
    
    def get_Nodes(self):
        return self.nodes

    def get_Weights(self):
        return self.weights
    
    # Should be deleted
    def resize(self, resize_factor = .5):
        self.nodes = self.nodes * resize_factor * self.n / np.sum(self.nodes)
        return self.nodes

    def randomizeWeights(self):
        self.weights = (2 * np.random.rand(self.n - self.input_len, self.n) - 1)
        self.zeroSelfLoops()
        return self.weights
    
    @staticmethod
    def shift_magnitude(r, min=0, max=1):
        print("shift: ", (min - max) * (r ** 2) + max)
        return (min - max) * (r ** 2) + max

    def randomWeightShift(self, magnitude=.05):
        self.weights = self.weights + (2 * np.random.rand(self.n-self.input_len, self.n) - 1) * .05
        self.zeroSelfLoops()
        return self.weights

    def zeroSelfLoops(self):
        self.weights *= np.ones((self.n - self.input_len, self.n)) - np.eye(self.n - self.input_len, self.n)
        return self.weights
        


    def draw(self):
        def clear_axis():
            ax.cla()     
            a = 1.4
            b = 1.4
            ax.axis('off')
            ax.set_xlim([-a, a])
            ax.set_ylim([-b, b])

        def update_graph():
            if 'graph' not in self.plt_data:
                G = nx.DiGraph()
                for i, val in enumerate(self.nodes):
                    G.add_node(i, value = val)
                for i in range(self.n - self.input_len, self.n):
                    G.add_node(i, value = self.last_stim[i - (self.n - self.input_len)])

                for i, row in enumerate(self.weights): # need to verify that this is consistent with architecture
                    for j, w in enumerate(row):
                        G.add_edge(j, i, weight = w)

                self.plt_data['graph'] = G
            
            else:
                G = self.plt_data['graph']
                for i, val in enumerate(self.nodes):
                    G.nodes[i]['value'] = val
                for i in range(self.n - self.input_len, self.n):
                    G.nodes[i]['value'] = self.last_stim[i - (self.n - self.input_len)]

                for i, row in enumerate(self.weights): # need to verify that this is consistent with architecture
                    for j, w in enumerate(row):
                        G.edges[j, i]['weight'] = w
            return self.plt_data['graph']


        def calculate_color(v):
            if v < -1.0:
                v = -1.0
            if v > 1.0:
                v = 1.0
            MIN_ALPHA = .1
            if -MIN_ALPHA < v < MIN_ALPHA:
                v = MIN_ALPHA
            return (POS_COLOR if v >= 0 else NEG_COLOR) + (abs(v),)

        def draw_nodes():
            border_colors = [OUTPUT_COLOR] * self.output_len + [HIDDEN_COLOR] * (self.n - self.input_len - self.output_len) + [INPUT_COLOR] * self.input_len
            fill_colors = [calculate_color(val) for val in list(nx.get_node_attributes(G, 'value').values())]
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=fill_colors, edgecolors=border_colors, node_size=node_size, linewidths=1.25)
            my_nx.draw_networkx_labels(G, pos, ax=ax, node_size=node_size)

            # Draw node values
            node_values = {k:round(v, 2) for k, v in nx.get_node_attributes(G, 'value').items()}
            nx.draw_networkx_labels(G, pos, ax=ax, labels=node_values, font_size=8)

        def draw_edges():

            edge_colors = {k:calculate_color(v) for k,v in nx.get_edge_attributes(G,'weight').items()}

            # Draw edges
            self_edges = [edge for edge in G.edges() if edge[0] == edge[1]]
            curved_edges = [edge for edge in list(set(G.edges()) - set(self_edges)) if reversed(edge) in G.edges()]
            straight_edges = list(set(G.edges()) - set(curved_edges) - set(self_edges))

            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges, edge_color = [edge_colors[k] for k in straight_edges], node_size=node_size)
            arc_rad = .075
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', edge_color = [edge_colors[k] for k in curved_edges], node_size=node_size)
            my_nx.draw_networkx_self_edges(G, pos, ax=ax, padding=SELF_LOOP_LENGTH, edgelist=self_edges, edge_color = [edge_colors[k] for k in self_edges] * 5, node_size=node_size)

            # Draw edge weights
            edge_weights = {k:round(v, 2) for k, v in nx.get_edge_attributes(G,'weight').items()}
            curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
            straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
            self_edge_labels = {edge: edge_weights[edge] for edge in self_edges}

            my_nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad, bbox=dict(facecolor='none', edgecolor='none'))
            nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False, bbox=dict(facecolor='none', edgecolor='none'))
            my_nx.draw_networkx_self_edge_labels(G, pos, ax=ax, padding=SELF_LOOP_LENGTH, node_size=node_size, edge_labels=self_edge_labels, rotate=False, bbox=dict(facecolor='none', edgecolor='none'))

        POS_COLOR = (1.0, 0.0, 0.0) # Red
        NEG_COLOR = (0.0, 0.0, 1.0) # Blue

        INPUT_COLOR = (0.0, 0.0, 0.0)
        HIDDEN_COLOR = POS_COLOR
        OUTPUT_COLOR = (0.0, 1.0, 0.0) # Green

        SELF_LOOP_LENGTH = 1.25


        if 'net_fig' not in self.plt_data:
            self.plt_data['net_fig'], ax = plt.subplots()
        else:
            ax = self.plt_data['net_fig'].get_axes()[0]
        
        G = update_graph() # G is just a reference to self.plt_data['graph']
        pos = nx.circular_layout(G)
        node_size = 300 * np.ones(self.n)

        clear_axis()
        draw_nodes()
        draw_edges()

        # print(np.round(self.weights, decimals=2))
        # print('*')
        # print(np.reshape(np.round(self.nodes, decimals=2), (-1, 1))) # need to append stim

    def draw_memory(self, range_=None):

        def dim(x):
            s = int(np.sqrt(x))
            dif = x - s * s
            if dif == 0:
                return s,s
            if dif <= s:
                return s+1, s
            return s+1, s+1            

        if range_ == None:
            range_ = range(self.n)

        if 'mem_fig' not in self.plt_data:
            (rows, cols) = dim(len(range_))
            self.plt_data['mem_fig'], axs = plt.subplots(rows, cols)
            self.plt_data['mem_fig'].tight_layout()
        elif len(range_) > len(self.plt_data['mem_fig'].get_axes()):
            (rows, cols) = dim(len(range_))
            pass # need to update axes of figure or delete fig and create a new one


        self.memory.draw_landscapes(self.plt_data['mem_fig'], range_)
                





        

