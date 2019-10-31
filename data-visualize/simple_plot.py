import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(-math.pi, math.pi, 0.1, dtype=float)
# print(x)

y = []
for i in range(len(x)):
    y.append(math.sin(x[i]))

# shorten version with numpy library
# y = np.sin(x)

# fig, ax = plt.subplots()
# print(type(ax))
# print(type(fig))
# ax.plot(x, y)
# ax.set_xlabel('X data')
# ax.set_ylabel('Y data')
# ax.set_title('y = sin(x) diagram')

plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.title('y = sin(x) diagram')
plt.xlabel('X data')
plt.ylabel('Y data')
# plt.show()

a = np.linspace(-math.pi, math.pi, 50, dtype=float)
# print(a)

b = []
for i in range(len(a)):
    b.append(math.cos(a[i]))

# shorten version with numpy library
b = np.cos(a)

plt.subplot(2, 1, 2)
plt.plot(a, b)
plt.title('a = sin(b)')
plt.xlabel('A data')
plt.ylabel('B data')
plt.show()
