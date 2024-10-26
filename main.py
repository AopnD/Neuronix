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
inputs = [1.2, 2.3, 3.4]
## the number of weights needs to match the number of biases otherwise you have incomplete list.
weights = [
    [0.7, -0.5, 1.25, 0.3],
    [0.8, -0.6, 1.3, -0.2],
    [0.6, -0.4, 1.2, 0.5],
    [-0.14, 0.2, -0.9, 0.7]
]
## theoraticly you can also use only biases or only weights, however this way make things a whole lot easier to calculate and change.
biases = [2, 13, 1.25, 4]

def get_output():
    if len(weights) != len(biases):
        print("Mismatch between weights and biases.")
        return
    results = []
    for i in range(len(weights)):
        weighted_sum = sum(inputs[j]*weights[i][j] for j in range(len(weights[i])))
        result = weighted_sum + biases[i]
        results.append(result)

    print(results)


get_output()