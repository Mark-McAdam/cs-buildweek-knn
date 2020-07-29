import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from collections import Counter
from matplotlib import style
from math import sqrt

# The magic distance formulae
# euclidean distance = sqrt( (x1-x2)**2 + (y1-y2)**2 )

# two mock classes and their features
dataset = {"k": [[1, 2], [2, 3], [3, 1]], "r": [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

# for i in dataset:
#   for ii in dataset[i]:
#     plt.scatter(ii[0], ii[1], s=100, color=i)

# in one line now for refactor
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100, color="yellow")
plt.show()
