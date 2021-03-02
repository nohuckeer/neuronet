import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


epochs = 200_000

trInputs = np.array([
    [0, 0, 1, 1],
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1]
])
# trInputs = np.array([
#     [0, 0, 1],
#     [1, 1, 1],
#     [1, 0, 1],
#     [0, 1, 1]
# ])
print("Training inputs:")
print(trInputs)

trOutputs = np.array([[0, 1, 1, 0]])
print("\nTraining outputs:")
print(trOutputs.T)

weights = np.random.random((4, 1))
print("\nRandom weights:")
print(weights)

outputs = sigmoid(np.dot(trInputs, weights))
print("\nOutputs:")
print(outputs)

print("\n")
print(outputs - trOutputs.T)

print("\n**** TRAINING ****")

for i in range(epochs):
    outputs = sigmoid(np.dot(trInputs, weights))
    err = trOutputs.T - outputs
    adj = np.dot(trInputs.T, (err * outputs * (1 - outputs)))
    weights += adj

print("\nWeights:")
print(weights)

print("\nOutputs:")
print(outputs)

print("\n**** TEST ****")
newInputs = np.array([0, 0, 0, 1])
outputs = sigmoid(np.dot(newInputs, weights))

print("\nOutputs:")
print(outputs)
