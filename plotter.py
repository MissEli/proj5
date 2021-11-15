import numpy as np
import matplotlib.pyplot as plt

Ec = []
file = np.loadtxt('ec.txt', dtype=str, skiprows=1)

for line in file:
    Ec.append(float(line))


# plot events per channel
x2 = np.arange(0, 200, 0.01)
plt.plot(x2,Ec,'-')
plt.xlabel('energy channel')
plt.ylabel('events')
plt.show()
