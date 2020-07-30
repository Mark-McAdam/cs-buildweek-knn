from MNN import MNN

k = MNN()


# using KNN native

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv("breast_cancer_data.txt")
df.replace("?", -99999, inplace=True)
df.drop(["id"], 1, inplace=True)

