import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(-math.pi, math.pi, 0.1, dtype=float)
# print(x)

y = []
for i in range(len(x)):
    y.append(math.sin(x[i]))

fig, ax = plt.subplots()
print(type(ax))
print(type(fig))
ax.plot(x, y)
ax.set_xlabel('X data')
ax.set_ylabel('Y data')
ax.set_title('y = sin(x) diagram')
plt.show()

a = np.linspace(-math.pi, math.pi, 50, dtype=float)
# print(a)

b = []
for i in range(len(a)):
    b.append(math.cos(a[i]))
