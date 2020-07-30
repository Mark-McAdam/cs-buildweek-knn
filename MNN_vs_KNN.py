""" 
Show time and accuracy comparison between My Nearest Neighbors 
and the Native SKLearn Nearest Neighbors

"""

# *************************
# MNN

mnn_accuracies = []
for i in range(25):
    # MY NN
    # TODO remove internal print statements in the function
    pass

    # this goes after the print("Accuracy") line
    mnn_accuracies.append(correct / total)

print("MNN Accuracy:")
print(sum(mnn_accuracies) / len(mnn_accuracies))


# *************************
# SKLearn KNN
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

knn_accuracies = []
for i in range(25):
    # k NN
    # TODO remove internal print statements in the function
    df = pd.read_csv("breast_cancer_data.txt")
    df.replace("?", -99999, inplace=True)
    df.drop(["id"], 1, inplace=True)

    X = np.array(df.drop(["class"], 1))
    y = np.array(df["class"])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.2
    )

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    # this goes after the print("Accuracy") line
    knn_accuracies.append(correct / total)

print("SKLearns KNN Accuracy:")
print(sum(knn_accuracies) / len(knn_accuracies))

