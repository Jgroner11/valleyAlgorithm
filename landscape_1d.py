import pyperclip
import math
import numpy as np
import matplotlib.pyplot as plt


class Peak:
    def __init__(self, c, a, s):
        self.c = c
        self.a = a
        self.s = s

    def val(self, x):
        return self.a * math.exp(-.5 * (x-self.c)**2 / self.s**2)

    def modify(self, x, strength):
        self.c = self.c + (x-self.c) / self.a
        # The final one in this equation is the max possible amplitude shift (Which occurs when the distance is 0)
        self.a = self.a + strength * (1 / (abs(x-self.c) + 1 / 1))
        self.s = .5


class Landscape:
    def __init__(self):
        self.peaks = []

    # Unused function, prob shoould be deleted
    def get_lists(self):
        all_cs = []
        all_as = []
        all_ss = []
        for p in self.peaks:
            all_cs.append(p.c)
            all_as.append(p.a)
            all_ss.append(p.s)
        return (all_cs, all_as, all_ss)

    def add_Peak(self, c, a, s):
        self.peaks.append(Peak(c, a, s))

        def get_c(n):
            return n.c

        self.peaks.sort(reverse=False, key=get_c)

    def calculate_threshold(self, range=(-10, 10)):
        return (range[1] - range[0]) * len(self.peaks) / 10.

    def update(self, x, strength):
        # For now, treat strentgh as if it is either -1 or 1
        closest_peak = self.find_closest_peak(x)
        if closest_peak != None and abs(x - closest_peak.c) < self.calculate_threshold():
            closest_peak.modify(x, strength)
        else:
            self.add_Peak(x, strength, .5)

    def find_closest_peak(self, x):
        amt_peaks = len(self.peaks)
        if amt_peaks == 0:
            return None
        if amt_peaks == 1 or x < self.peaks[0].c:
            return(self.peaks[0])
        lower = self.peaks[0]
        for i in range(len(self.peaks) - 1):
            upper = self.peaks[i+1]
            if(lower.c <= x < upper.c):
                if abs(lower.c-x) <= abs(upper.c - x):
                    return lower
                else:
                    return upper
            lower = upper
        return(upper)

    def clean(self, threshold=.1):
        self.peaks[:] = [x for x in self.peaks if x.a >= threshold]

    def f(self, x):  # x can be single value or array
        def f_single(acc, peaks):
            if len(peaks) == 0:
                return acc
            else:
                n = peaks.pop()
                return f_single(acc+n.val(x), peaks)

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

    def plot(self, start=-10, end=10, num_points=100, axes=(-10, 10, -10, 10)):
        x = np.arange(num_points) * (end - start) / (num_points - 1) + start
        y = np.array(self.f(x.tolist()))
        plt.axis(axes)
        plt.plot(x, y)
        plt.show()

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

    def update_plot(self, fig, curve):
        curve.set_ydata(self.f(curve.get_xdata()))
        fig.canvas.draw()
        fig.canvas.flush_events()

    def to_desmos(self):
        out = "y=0"
        for p in self.peaks:
            out += "+" + str(p.a) + \
                "e^{-\\frac{\left(x-" + str(p.c) + \
                "\\right)^{2}}{2\left(" + str(p.s) + "^{2}\\right)}}"
        print("Copying the following to clipboard:\n" + out)
        pyperclip.copy(out)
        return out
