import numpy as np
from engine.losses import MSE
from engine.shapes import Linear
from engine.shapes import Tensor
from engine.optimizers import SGD

# Set a seed so that we get reproducible results
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layer1 = Linear(input_dim, output_dim, "Layer 1")

    def forward(self, input): 
        self.input = input
        return self.layer1.forward(input)#.relu()

    def backward(self, grad):
        layer1Grad = self.layer1.backward(grad)
        
    def parameters(self):
        return [self.layer1.weights, self.layer1.biases]

nn = NeuralNetwork(1, 0, 1)
sgd = SGD(nn.parameters())
mse = MSE()

np.random.seed(0)
# Create 1000 random input values between -10 and 10
inputs = np.random.uniform(-10, 10, size=(1000, 1))
# Compute the corresponding target values, adding a small amount of random noise
targets = 2 * inputs + 1 + np.random.normal(scale=0.5, size=(1000, 1))
dataset = list(zip(inputs, targets)) # makes iterateable 

for epoch in range(200):  
    for input, target in dataset:               # dataset is an iterable of (input, target) pairs
        input =  input.reshape(-1, 1).T
        target = target.reshape(-1, 1).T
        target = Tensor(target)                 # converts target data into a tensor        
        inputTensor  = Tensor(input.reshape(1,-1))
        prediction = nn.forward(inputTensor)    # performs a forward pass on the neural network
        loss = mse.forward(prediction, target)  # calculates loss on the prediction using MSE function
        gradient = mse.backward()               # calculates gradient by performing backprop - input is the gradient of the loss func
        nn.backward(gradient)                   # performs backprop on neural network - input is the gradient of the loss wrt the networks output (descending to low loss)
        sgd.step()                              # updates the network parameters using the gradients calculated in backprop

print("Done")

print(nn.layer1.weights.values)
print(nn.layer1.biases.values)
