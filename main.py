from landscape_1d import Landscape
from matplotlib import pyplot as plt
import numpy as np

ls1 = Landscape()
# ls1.add_Peak(-3., 1.2, .5)
# ls1.add_Peak(2., 1., .5)


ls1.to_desmos()

fig, curve = ls1.plot_int()
while True:
    
    inp = [float(i) for i in input().split()]
    ls1.update(inp[0], inp[1])
    ls1.update_plot(fig, curve)
