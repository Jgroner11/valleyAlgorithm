from memory import Landscape
from net import Net
from netExperiment import NetExperiment
from environment import BasicEnv

from matplotlib import pyplot as plt
import numpy as np




def test_landscape_deprecated():
    ls1 = Landscape()
    fig, curve = ls1.plot_int()
    while True:
        for peak in ls1.peaks:
            print("(" + str(round(peak.c, 1)), peak.a, str(round(peak.s, 1)), str(round(peak.o, 1)) + ") ")
        inp = [float(i) for i in input().split()]
        ls1.update(inp[0], inp[1])
        ls1.to_desmos()
        ls1.update_plot(fig, curve)

def test_landscape():
    plt.ion()
    ls1 = Landscape()
    fig, ax = plt.subplots(1, 1)
    while True:
        ls1.draw(ax, axes=(0, 1, -3, 3), num_points=1000)
        # for peak in ls1.peaks:
        #     ax.text(peak.c, ls1.f(peak.c), round(peak.o, 1), fontsize=8, ha='center')

        ls1.to_desmos(suppress_text=False)
        for peak in ls1.peaks:
            print("(" + str(peak.c), peak.a, str(peak.s), str(peak.o) + ") ")
        inp = [float(i) for i in input().split()]
        ls1.update(inp[0], inp[1])


def test_experiment():
    exp = NetExperiment.read_csv("test1_2")
    net = exp.to_net(2)
    test_net(net)

def generate_data(name, range, num_trials, resize):
    for i in range:
        exp = NetExperiment(i, name=name+str(i), num_trials=num_trials, resize=resize)
        exp.run()
        exp.write_csv()

def hist_data(file_list):
    num_plots = len(file_list)
    fig, axs = plt.subplots(num_plots, 1)

    n_values = np.array(range(num_plots)) + 2
    for i, name in enumerate(file_list):
        exp = NetExperiment.read_csv(name)
        exp.hist(axs[i])
    axs[-1].set_xlabel('Cycle Length')
    axs[0].set_title('Cycle Distribution Over Different n')
    # axs[0].set_title('Cycle Distribution Without 0 and 1')
    plt.show()
        
def get_name_list(name, range):
    lst = []
    for i in range:
        lst.append(name + str(i))
    return lst

def test_net(net=None):
    if net == None:
        net = Net(5, input_len= 0, output_len=1)
    plt.ion()
    while True:
        net.draw_memory()
        print(np.sum(net.nodes))
        s = input()
        if s == "":
            net.step()
        elif s == "x":
            net.randomWeightShift()
        else:
            for i in range(int(s)):
                net.step()

def test_basic_net():
    num_success = 0
    for i in range(500):
        env = BasicEnv()
        net = Net(5, 0, 1)
        num_success += env.run(net, iters=500)
    print(num_success / 500)


# generate_data('no_resize_selfLoops_', range(2, 50), 1000, False)
# hist_data(get_name_list('no_resize_selfLoops_', range(6, 48, (48-6) // 4)))

BasicEnv().run(Net(10, 0, 1))



# test_landscape()
