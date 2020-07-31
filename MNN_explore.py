import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
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
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=100, color="yellow")
# plt.show()

# TODO - refactor this into fit and predict methods

# input of x
# input of y
# defau;t k value of 3 which can be changed
# TODO incorporate k = (len(data)+1) if (len(data)%2==0) else len(data)+2 below


def M_N_N(x, y, k=3):

    # if user tries to set k less than total voting groups warn user
    if len(x) >= k:
        warnings.warn(
            "set k value to less than total voting groups\nthis can lead to ties in voting"
        )

    # need to compare our prediction
    # distance of prediction to all other data points
    # then pick the data points closest using euclidean distance

    # instantiaite empty list to hold the distances
    distances = []

    # for group in x
    for group in x:
        # for features in x's group
        for features in x[group]:
            # calculate euclidean distance
            # euclidean distance formula - sqrt( (x1-x2)**2 + (y1-y2)**2 )

            # solution for a 2d
            # euclidean_distance = sqrt ( (features[0]-y[0])**2 + (features[1]-y[1])**2 )

            # solution for multidimensional array
            # this allows the algorithm to scale to dataset with more than 2 features
            # use of np array allows for higher dimensionality
            euclidean_distance = np.sqrt(
                np.sum((np.array(features) - np.array(y)) ** 2)
            )

            # this can be refactored to use the np.linalg.norm() function
            # ^ calculates euclidean distance.

            # TODO implement this for refactor later after I time and log results from
            # solution for multidimensional array
            # this looks hard to read
            # euclidean_distance = np.linalg.norm(np.array(features)-np.array(y))

            # append the distances to a list of lists that include the group
            distances.append([euclidean_distance, group])

    # votes is keeping track of which group most voted for
    # later will count the most voted for group
    # i[o] distance, i[1] is the group it belongs to
    votes = [i[1] for i in sorted(distances)[:k]]

    # vote_result is the most voted for
    # most common returns array of list so index to the good stuff
    vote_result = Counter(votes).most_common(1)[0][0]

    # confidence is how sure that the result is in the right group
    # if 100% of the votes are for a group there is higher confidence in accuracy
    # number of votes divided by number of groups k gives confidence percentage
    confidence = Counter(votes).most_common(1)[0][1] / k

    # vote_result returns a list of a tuple ie. [('r',3)]
    # print(Counter(votes).most_common(1))

    # print(vote_result)
    # print(confidence)
    return vote_result, confidence


# practice on a single item.
# result, confidence = K_N_N(dataset, new_features, k=3)
# print(result, confidence)


# lets graph this up a little bit
# star on the predicted point - colored in the group it belongs
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=70, color=result, marker="*")
# plt.show()


# /Users/mark/Desktop/cs-buildweek-knn/breast_cancer_data.txt
# breast_cancer_data.txt
# 11. Class:                        (2 for benign, 4 for malignant)


df = pd.read_csv("data/breast_cancer_data.txt")
df.replace("?", -99999, inplace=True)
df.drop(["id"], 1, inplace=True)

# there is a column of string that represent integer
# converting all value to float will help future versions
full_df = df.astype(float).values.tolist()

# since full_df is a list of list we can shuffle to order to randomize
random.shuffle(full_df)

# test size for train test split
test_size = 0.2

# train and test set both have two items
# first is the list of cases with class 2 Benign
# second item is the list of cases with Class 4 Malignant
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}

# train data is everything up to the last 20% of the data
train_data = full_df[: -int(test_size * len(full_df))]
# test data is the last 20% of the data
test_data = full_df[-int(test_size * len(full_df)) :]

# iterate through train data
# each item i is a list
# the last item in the list is the tumor class 2benign 4 malignant
for i in train_data:
    train_set[i[-1]].append(i[:-1])


# iterate through test data
# each item i is a list
# the last item in the list is the tumor class 2benign 4 malignant
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = M_N_N(train_set, data, k=5)
        if group == vote:
            correct += 1
        else:
            print(confidence)
        total += 1
print("Accuracy:", correct / total)

