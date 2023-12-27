import pyperclip
import math
import bisect
import numpy as np
import matplotlib.pyplot as plt
import my_matplotlib as my_plt


# important functions:
#   Calculate threshold, calculates max distance a point can be away from a peak before addding a new peak
#   Clean -> removes small peaks,  in future could merge peaks
#   Modify
#       Idea - modify sd relative to a such that max derivative is the distance from position of max derivative to center


class Peak:
    '''
    Object representing a gaussian function
    a: amplitude
    c: center
    s: standard deviation
    o: ossification -> How resilient the peak is to changes
    '''
    O_MAX = 50.0
    O_MIN = 1.0


    def __init__(self, c, a):
        self.c = c
        self.a = Peak.__unzero(a)
        self.s = .1 # Currently, all standard deviations are .1 for all peaks
        self.o = Peak.O_MIN

    def f(self, x):
        return self.a * math.exp(-.5 * (x-self.c)**2 / self.s**2)
    
    def f_prime(self, x):
        return self.f(x) * (self.c-x) / self.s**2

    def modify(self, x, r):
        '''
        Modifies peak based on new point (x, r)
        '''
        
        c_inc =  (x-self.c) / (self.o + math.e ** -(abs(r) - abs(self.a)))
        c_new = self.c + c_inc

        # The final one in this equation is the max possible amplitude shift multiplier (Which occurs when the distance is 0)
        a_new = Peak.__unzero(self.a + (r - self.a) / (self.o + 1.0))
        # * (1 / (abs(x-self.c) + 1 / 1))

        o_inc = self.__calculate_o_inc(self.a, r)
        o_new = max(Peak.O_MIN, min(self.o + o_inc, Peak.O_MAX))

        self.c = c_new
        self.a = a_new
        self.o = o_new

    @staticmethod
    def __unzero(x, zero_threshold = .001):
        if 0 <= x < zero_threshold:
            return zero_threshold
        elif -zero_threshold < x < 0:
            return -zero_threshold
        return x
        


    @staticmethod
    def __calculate_o_inc(a, r):
        '''
        Returns the amount to modify the ossification value (o) by based on the distance between the peak and the reward
        a: peak ampltitude
        r: current reward
        https://www.desmos.com/calculator/sycotso3o6
        '''

        # The max value o can be decreased by. Note that the max o_inc is 1.
        # Must have DEC_MAX > 1
        # asymptotes of function are going to be y=DEC_MAX and y=-DEC_MAX
        DEC_MAX = 1.5

        # distance between peak and reward
        d = abs(a - r)

        # Sigmoid function becomes super steep if a is too small. To fix this, if a is smaller than .1, just pick 1 or DEC_MAX
        if abs(a) < .1:
            if d < .1:
                return 1
            return -DEC_MAX
        else:
            k = (DEC_MAX - 1) / (DEC_MAX + 1)
            b = k ** (-2 / abs(a)) # this value gets very big at small a
            return (2 * DEC_MAX) / (1 + k * b ** d) - DEC_MAX

    @staticmethod
    def combine(peak1, peak2):
        return Peak((peak1.c + peak2.c) / 2, (peak1.a + peak2.a) / 2)

class Landscape:
    '''
    Object representing a sum of gaussian functions
    peaks: list of Peak objects sorted by c
    '''

    def __init__(self):
        self.peaks = []

    def add_Peak(self, c, a):
        bisect.insort(self.peaks, Peak(c, a), key=lambda x: x.c)

    # Replaced by X_THRESH
    # def calculate_threshold(self, range=(0, 1)):
    #     '''
    #     Calculates the distance a new input must be away from existing peaks in order to cause a new peak to be created
    #     Range (tuple): range of landscape, should be the same as the potential values the neuron can be
    #     '''
    #     return .1
    #     return (range[1] - range[0]) / 10 # threshold = .1, at most 10 peaks
    #     return (range[1] - range[0]) * len(self.peaks) / 20. #The more peaks, the harder it is to make a new one (bigger threshold). The last number (20) represents the max number of peaks which can exist within the range I think

    def update(self, x, strength, min_gap=.1):
        closest_peak = self.find_closest_peak(x)
        if closest_peak != None and abs(x - closest_peak.c) < min_gap:
            closest_peak.modify(x, strength)
        else:
            bisect.insort(self.peaks, Peak(x, strength), key=lambda x: x.c)

    def find_closest_peak(self, x):
        '''
        Returns the peak whose center is closest to x
        If there are no peaks, returns None
        '''
        def get_dist(index):
            if 0 <= index < len(self.peaks):
                return abs(self.peaks[index].c - x)
            return float('inf')
        
        b = bisect.bisect(self.peaks, x, key=lambda x: x.c)
        diff = get_dist(b) - get_dist(b-1)

        if diff >= 0:
            return self.peaks[b-1]
        if diff < 0:
            return self.peaks[b]
        return None # in the case where there no peaks, diff will be nan, thus neither of the above conditions will be satisfied

    def clean(self, min_a=.01, min_gap=.05):
        '''
        Removes all peaks whose amplitude(a) is less than min_a 
        Combines all peaks whose centers(c) are less than min_gap apart
        Time Efficiency: O(n)
        Maintains peaks invariant
        '''

        # Works by iteratively popping peaks from the back of self.peaks, ignoring any peak whose amplitude value is less than min_a. 
        # It stores the most recently popped peaks in p1 and p2 (p2 being the more recently popped peak and thus the one with the smaller c value). 
        # If p1 and p2 are close to each other (relative to min_gap), it combines them and stores that value in p1, 
        # otherwise p1 is clean and can be added to the list of cleaned peaks. 
        # Since peaks are added to the new list from highest c value to lowest, in order to maintain the invariant, the order of the peaks peaks must be reversed.

        def pop_clean_peak():
            if self.peaks:
                p = self.peaks.pop()
                if abs(p.a) >= min_a:
                    return p
                return pop_clean_peak()
            return None

        peaks_cleaned = []

        p1 = pop_clean_peak()
        if p1 is not None:
            p2 = pop_clean_peak()
            while p2 is not None:
                if p1.c - p2.c < min_gap:
                    p1 = Peak.combine(p1, p2)
                else:
                    peaks_cleaned.append(p1)
                    p1 = p2
                p2 = pop_clean_peak()
            peaks_cleaned.append(p1)

        peaks_cleaned.reverse()
        self.peaks = peaks_cleaned

    def f(self, x):  # x can be single value or array
        def f_single(acc, peaks):
            if len(peaks) == 0:
                return acc
            else:
                n = peaks.pop()
                return f_single(acc+n.f(x), peaks)

        def f_list():
            out = []
            for v in x:
                out.append(self.f(v))
            return out
        
        if isinstance(x, list):
            return f_list()
        elif isinstance(x, np.ndarray):
            return self.f(x.tolist())
        else:
            return f_single(0, list(self.peaks))
        
    def f_prime(self, x):  # x can be single value or array
        def f_prime_single(acc, peaks):
            if len(peaks) == 0:
                return acc
            else:
                n = peaks.pop()
                return f_prime_single(acc+n.f_prime(x), peaks)

        def f_list():
            out = []
            for v in x:
                out.append(self.f_prime(v))
            return out
        
        if isinstance(x, list):
            return f_list()
        elif isinstance(x, np.ndarray):
            return self.f_prime(x.tolist())
        else:
            return f_prime_single(0, list(self.peaks))        

    def plot(self, start=-10, end=10, num_points=100, axes=(-10, 10, -10, 10)):
        x = np.arange(num_points) * (end - start) / (num_points - 1) + start
        y = np.array(self.f(x.tolist()))
        plt.axis(axes)
        plt.plot(x, y)
        plt.show()

    # interactive plot, deprecated, should use draw instead
    def plot_int(self, start=-10, end=10, num_points=100, axes=(-10, 10, -10, 10)):
        x = np.arange(num_points) * (end - start) / (num_points - 1) + start
        y = np.array(self.f(x.tolist()))
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if isinstance(axes, tuple):
            ax.set_xlim(axes[0], axes[1])
            ax.set_ylim(axes[2], axes[3])
        curve, = ax.plot(x, y, 'r-')
        fig.canvas.draw()
        fig.canvas.flush_events()
        return (fig, curve)
    
    def draw(self, ax, num_points=100, start=0, end=1):

        x = np.arange(num_points) * (end - start) / (num_points - 1) + start
        y = np.array(self.f(x.tolist()))

        for peak in self.peaks:
            ax.text(peak.c, self.f(peak.c), round(peak.o, 1), fontsize=8, ha='center')
        ax.plot(x, y, 'r-')

    # deprectated, should use draw instead
    def update_plot(self, fig, curve):
        curve.set_ydata(self.f(curve.get_xdata()))
        fig.canvas.draw()
        fig.canvas.flush_events()

    def to_desmos(self, suppress_text=False):
        out = "y=0"
        for p in self.peaks:
            out += "+" + str(p.a) + \
                "e^{-\\frac{\left(x-" + str(p.c) + \
                "\\right)^{2}}{2\left(" + str(p.s) + "^{2}\\right)}}"
        if not suppress_text:
            print("Copying the following to clipboard:\n" + out)
        pyperclip.copy(out)
        return out

class Memory:
    def __init__(self, n):
        self.n = n
        self.landscapes = np.array([Landscape() for _ in range(n)])
        self.clean_counter = 0
        
        # These two variables store all past brain states directly, eventually want to remove this bc all required info about past states will be emulated in landscapes
        self.past_states = list()
        self.past_rewards = np.array([])

    def update(self, nodes, r, clean_freq=25):
        '''
        nodes : np.array, values of nodes
        r : int, reward level
        '''

        for i in range(self.n):
            self.landscapes[i].update(nodes[i], r)

        if self.clean_counter == clean_freq:
            self.landscapes[i].clean()
            self.clean_counter = 0
        else:
            self.clean_counter += 1

        self.past_states.append(nodes)
        self.past_rewards = np.append(self.past_rewards, r)
        
    def compute_derivatives(self, nodes):
        derivatives = np.zeros(self.n)
        for i in range(self.n):
            derivatives[i] = self.landscapes[i].f_prime(nodes[i])
        # print(derivatives)
        return derivatives
        
    
    def gradient(self, nodes):
        return np.array([self.landscapes[i].f_prime(nodes[i]) for i in range(self.n)])
    
    def draw_landscapes(self, fig, range, nodes, axes=(0, 1, -4, 4)):
        axs = fig.get_axes()
        for i, node in enumerate(range):
            axs[i].cla()
            if isinstance(axes, tuple):
                axs[i].set_xlim(axes[0], axes[1])
                axs[i].set_ylim(axes[2], axes[3])            

            for j, state in enumerate(self.past_states):
                axs[i].scatter(state[node], self.past_rewards[j], c='blue', alpha=(j / len(self.past_states)), marker='.')

            c = self.landscapes[node].f_prime(nodes[node])
            x = nodes[node]
            y = self.landscapes[node].f(nodes[node])
            dx = 1 / math.sqrt(c ** 2 + 1)
            dy = c / math.sqrt(c ** 2 + 1)

            # When the aspect ratio does not equal 1, a line segment of true distance 1 has a different on screen distance depending on the slope
            # To combat this, we compute the onscreen distance in terms of the units of the x-axis in order to keep the onscreen distance constant
            r = my_plt.get_aspect(axs[i])
            dox = math.sqrt(dx**2 + (dy* r)**2)

            # Current x position
            axs[i].plot([x, x], [axes[2], axes[3]], c = 'green', linestyle='-')
            
            # Strength of derivative
            WEIGHT_SHIFT_MAG = .025 
            axs[i].arrow(x, axes[3] - 1, WEIGHT_SHIFT_MAG * c, 0, head_width=.15, head_length=.01, length_includes_head = True, fc='green', ec='green')
            
            # memory function
            self.landscapes[node].draw(ax=axs[i])

            # directional derivative
            GRADIENT_SCALE_FACTOR = .075 
            axs[i].annotate("", xy=(x + GRADIENT_SCALE_FACTOR * dx / dox, y + GRADIENT_SCALE_FACTOR * dy / dox), xytext=(x, y), arrowprops=dict(arrowstyle="->", shrinkA=0))

            axs[i].set_title('' + str(node), pad=0, loc='center')

