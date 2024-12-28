import numpy as np

# Representation of a single node with 3 inputs ( data we getr from other nodes\ training data)
# weights and biases pre-determined for now.

# inputs = [1.2, 2.3, 3.4]
# weights = [0.7, -0.5, 1.25]
# bias = 2
#
# output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
#
# print(output)

# Making a representation on multiple nodes in the output layer, resaving the same inputs but with different Weights
# and Biases

## here we have 4 neurons, resiving 3 inputs each, and returning a list of 4 new predictions from the output layer.
# inputs = [1.2, 2.3, 3.4, 4.5]
## the number of weights needs to match the number of biases otherwise you have incomplete list.
# weights = [
#     [0.7, -0.5, 0.25, 0.3],
#     [0.8, -0.6, 1.3, -0.2],
#     [0.6, -0.4, 0.62, 0.5],
#     [-0.14, 0.2, -0.9, 0.7]
# ]
## theoraticly you can also use only biases or only weights, however this way make things a whole lot easier to calculate and change.
# biases = [2, 1.3, 1.25, 1]

# def get_output():
#     if len(weights) != len(biases):
#         print("Mismatch between weights and biases.")
#         return
#     results = []
#     for i in range(len(weights)):
#         weighted_sum = sum(inputs[j] * weights[i][j] for j in range(len(weights[i])))
#         result = weighted_sum + biases[i]
#         results.append(result)
#
#     print(results)
#
#
# get_output()


## Shapes
## At each dimension, what's the size of that dimension

## One-dimensional list:
## Shape: (1, 4) Type: 1D Array, Vector.
# l = [7, 4, 2, 6]

## Two-dimensional list:
## Shape: (2, 4) Type: 2D Array, Matrix.
# lol = [
#     [1, 2, 3, 4],
#     [5, 6, 7, 8]
# ]

## Three-dimensional list:
## Shape: (3, 3, 5) Type: 3D Array.
# lolol = [
#     [
#         [1, 2, 3, 4, 5],
#         [6, 7, 8, 9, 10],
#         [1.1, 1.2, 1.3, 1.4, 1.5]
#     ],
#     [
#         [5, 4, 3, 2, 1],
#         [10, 9, 8, 7, 6],
#         [2.1, 2.2, 2.3, 2.4, 2.5]
#     ],
#     [
#         [8, 2, 5, 1, 9],
#         [1.7, 6, -12, 4, 10],
#         [3.1, 3.2, 3.3, 3.4, 3.5]
#     ]
# ]

## What dot_product does:
# Example with two 1D vectors:
# a= [1,2,3]
# b = [2,3,4]
# dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
# print(dot_product)

## np.dot with two 1D vectors:
# Explanation: This example calculates the dot product of two 1D arrays.
# inputs = [1.2, 2.3, 3.4, 4.5]
# weights = [0.7, -0.5, 0.25, 0.3]
# bias = 2
# output = np.dot(weights, inputs) + bias
# print(output)


## np.dot with weights as a matrix:
# Explanation: This example shows matrix-vector multiplication using np.dot.
# inputs = [1.2, 2.3, 3.4, 4.5]
# weights = [
#     [0.7, -0.5, 0.25, 0.3],
#     [0.8, -0.6, 1.3, -0.2],
#     [0.6, -0.4, 0.62, 0.5],
#     [-0.14, 0.2, -0.9, 0.7]
# ]
# biases = [2, 1.3, 1.25, 1]
# output = np.dot(weights, inputs) + biases
# print(output)



# inputs is now a batch of input vectors, represented as a matrix with the shape (3,4),
# where each row corresponds to an input vector, and each row has 4 elements (floats).
# this is essentially a "list of lists" (lol), with 3 lists inside and each containing 4 floats.

# weights is a matrix of shape (3,4), where each row corresponds to the weights for one neuron.
# to compute the dot product, we transpose weights into a shape of (4,3), so that
# the second dimension of inputs matches the first dimension of weights (after transposition).

#  **more detailed explanation of batches and their significance could be added here.



inputs = [
    [1.2, 2.3, 3.4, 4.5],
    [2.2, 3.3, 4.4, 5.5],
    [3.2, 4.8, 5.4, 6.5]
]
weights = [
    [0.7, -0.5, 0.25, 0.3],
    [0.8, -0.6, 1.3, -0.2],
    [0.6, -0.4, 0.62, 0.5],
]
biases = [2, 1.3, 1.25, 1]


outputs = np.dot(inputs, np.array(weights).T) + biases
print(outputs)