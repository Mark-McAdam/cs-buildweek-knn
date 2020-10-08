# My Nearest Neighbors 
- Implementation of Scikit-Learn K-Nearest Neighbors from scratch in python. 

## Table of Contents
- [Description](#description)
- [Functionality](#functionality) 
- [Technology](#technology)
- [License](#license)

## Description 
- My Nearest Neighbors (MNN) all live closest to me

Closest to me in terms of euclidean distance. Cosine similarity can be useful to measure the angle between vectors, while disregarding magnitude and weight. Euclidean distance measure the distance between points. 

Distance is easy to conceptualize when thinking in terms many can relate to, imagine a neighborhood. Homes are basically aligned along a grid. X and Y coordinates, with the possibility of neighbors in front behind and on both sides. Those neighbors can have neighbors arranged in a similar fashion to their home. 

Distance to my neighbors would be harder to calculate if I lived in an apartment building or condos. I would have another dimension to measure distance along. The problem becomes more difficult when more dimensions are introduced. Take all coordinates of x and y and multiply for z possibilities to account for the floors above and potentially below the starting point. 

Perhaps the nearest neighbor lived in the space you reside but in the past, or your closes neighbor is now perhaps the person who lives there after you. We have introduced time as a new dimension now. Distance abstracted farther than physical dimensions is where my analogy starts to fall apart.  Consider that the complexity of the problem increases as the dimensionality of the problem does. 

Why does dimensionality matter ? The dataset I am working with has 9 features. 
- clump_thickness,
- uniform_cell_size,
- uniform_cell_shape,
- marginal_adhesion,
- single_epi_cell_size,
- bare_nuclei,
- bland_chromation,
- normal_nucleoli,
- mitoses

class - 2 Benign 4 Malignant

Each column is a feature, each row is an observation, and finally also the class which is the target variable to be predicted. 

## Functionality
Brute force approach is what I am writing. We can compare with SKlearns brute force version as well. The default algorithm choice of 'auto' attempts to look at the data and decide on the best algorithm to use. For this reason Sklearn's library will oftentimes be faster than attempting to rewrite everything from scratch. 

One shortcut that Sklearn can use, is to only measure distance to neighbors within a radius from the point we are trying to predict. This narrows down the runtime and complexity of the problem. 

## Technology
What are the technologies used?

- Python
- Numpy
- Scikit-Learn for comparison


## License
[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/Mark-McAdam/any-nlp/blob/master/LICENSE)
