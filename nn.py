from engine.losses import MSE
from engine.shapes import Linear
from engine.shapes import Tensor
from engine.optimizers import SGD
import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layer1 = Linear(input_dim, hidden_dim, "Layer 1")
        self.layer2 = Linear(hidden_dim, output_dim, "Layer 2")

    def forward(self, input): 
        self.input = input
        hidden = self.layer1.forward(input).relu()
        return self.layer2.forward(hidden)

    def backward(self, grad):
        layer2Grad = self.layer2.backward(grad)
        #print("layer2 gradients: " + str(layer2Grad))
        layer1Grad = self.layer1.backward(layer2Grad)
        #print("layer1 gradients: " + str(layer1Grad))

    # add the following to your NeuralNetwork class
    def parameters(self):
        return [self.layer1.weights, self.layer1.biases, self.layer2.weights, self.layer2.biases]

# example of a training loop
nn = NeuralNetwork(10, 20, 1)
mse = MSE()
sgd = SGD(nn.parameters())

def print_graph(output):
    print(output.shape())

    if(output.creator_op is not None):
        print(output.creator_op)
        for c in output.creators:
            print_graph(c)
    return


np.random.seed(0)
inputs = np.random.randn(100, 10)  # 100 samples, 10 features each
targets = np.random.randn(100, 1)  # 100 samples, 1 target value each

dataset = list(zip(inputs, targets)) # makes iterateable 

#input = Tensor(inputs[0].reshape(-1, 1).T)
#target = Tensor(targets[0].reshape(-1, 1))
'''
output = nn.forward(input)

print("target:")
print(target.values)

print("loss:")
#takes two tensors as input 
loss = mse.forward(output,target)
print(loss)

print("gradient:")
gradient = mse.backward()
print(gradient)
print(gradient.shape)

print("backward stage")
nn.backward(gradient)

#print("layer 1 grads:")
#print(nn.layer1.weights.grads)

print("sgd stage:")

print("bias shape")
print(nn.layer1.biases.shape())
sgd.step()

'''
for epoch in range(3):  # 100 epochs
    print(epoch)
    for input, target in dataset:               # dataset is an iterable of (input, target) pairs
        input =  input.reshape(-1, 1).T
        target = target.reshape(-1, 1).T
        target = Tensor(target)                 # converts target data into a tensor        
        inputTensor  = Tensor(input.reshape(1,-1))
        prediction = nn.forward(inputTensor)    # performs a forward pass on the neural network
        loss = mse.forward(prediction, target)  # calculates loss on the prediction using MSE function
        gradient = mse.backward()               # calculates gradient by performing backprop - input is the gradient of the loss func
        #print(prediction.creator_op + " " + str(prediction.creators))
        nn.backward(gradient)                   # performs backprop on neural network - input is the gradient of the loss wrt the networks output (descending to low loss)
        print("loss: " + str(loss))
        
        sgd.step()                              # updates the network parameters using the gradients calculated in backprop

print("Done")
