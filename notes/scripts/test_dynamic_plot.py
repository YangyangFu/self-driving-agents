import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')
while True:
    xdata.append(np.random.rand(1))
    ydata.append(np.random.rand(1))
    ln.set_data(xdata, ydata)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.1)
    if len(xdata) > 100:
        break