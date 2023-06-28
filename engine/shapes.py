import numpy as np

class Tensor:
    def __init__(self, values, creators = None, creator_op = None):
        self.values = values
        self.grads = None
        self.creators = creators #the tensors that made this tensor
        self.creator_op = creator_op
    
    def __add__(self, other):
        try:
            return self.__class__(self.values + other.values,creators=[self,other],creator_op="add")
        except ValueError:
            print("shapes are not aligned for matrix addition")
            return
        
    def __sub__(self, other):
        try:
            return self.__class__(self.values - other.values,creators=[self,other],creator_op="sub")
        except ValueError:
            print("shapes are not aligned for matrix addition")
            return

    def __mul__(self, other):
        #check that other is of the Tensor class before trying to mul
        if isinstance(other, (int,float)):
            result = self.values * other
            return self.__class__(result,creators=[self, other], creator_op="mul")
        if isinstance(other, self.__class__):
            try:
                return self.__class__(np.matmul(self.values, other.values), creators=[self,other], creator_op="mul")
            except ValueError:
                print("shapes are not aligned for matrix multiplication") 
            return
        
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            result = self.values ** other
            return self.__class__(result, creators=[self, other], creator_op="pow")
        elif isinstance(other, Tensor):
            result = self.values ** other.values
            return self.__class__(result, creators=[self, other], creator_op="pow")
        else:
            raise TypeError("`other` must be a scalar or a Tensor")

    def __div__(self, other):
        if isinstance(other, (int, float)):
            result = self.values / other
            return self.__class__(result, creators=[self], creator_op="div")
        elif isinstance(other, Tensor):
            result = self.values / other.values
            return self.__class__(result, creators=[self, other], creator_op="div")
        else:
            raise TypeError("`other` must be a scalar or a Tensor")
        
    def relu(self):    
        return self.__class__(self.values * (self.values > 0), creators=[self], creator_op="relu") 
    
    def T(self):
        return self.__class__(self.values.T, creators=[self], creator_op="transpose")

    def backward(self, grad=None):
        #print("tensor backward")
        '''
        Peforms the backward pass on the computation graph
        The nodes are Tensor objects and the edges are operations, information on the operations is 
        stored in creators and creator_op

        1. If the gradient is not provided at the current node the backward pass is just starting at this node
           therefore, we need to initialise the gradient as an array of 1s
        2. Depeding on the creator_op used to create the current node, you will compute the grad
           at the nodes that were used to create the current node (creators).
           For some operations the gradient doesnt change, in this case the gradient is passed unchanged
        3. After computing the gradients at the creators you call backward on each creator, providing them with
           them with their corresponding gradients
        4. Keeping track of the gradients computed at each node for later use using the grads attribute
        '''

        # For input tensors (no creator op), don't need to perform any backward computation.
        if self.creator_op is None:
            if grad is None:
                self.grads = np.ones_like(self.values)
            else:
                self.grads = grad
            return

        # When back propagation just started, create an initial gradient of ones with the same shape.
        if grad is None:
            self.grads = np.ones_like(self.values)
        else:
            self.grads = grad

        # If tensor was created by an addition operation, the gradient doesn't change.
        if self.creator_op == "add":
            self.creators[0].backward(self.grads)
            self.creators[1].backward(self.grads)

        # If tensor was created by a multiplication operation, gradients change according to chain rule.
        if self.creator_op == "mul":
            self.creators[0].backward(self.grads @ self.creators[1].values.T)
            self.creators[1].backward(self.creators[0].values.T @ self.grads)

        # If tensor was created by a subtraction operation.
        if self.creator_op == "sub":
            self.creators[0].backward(self.grads)
            self.creators[1].backward(-self.grads)

        # If tensor was created by a power operation.
        if self.creator_op == "pow":
            new_grad = self.creators[1] * (self.creators[0].values ** (self.creators[1] - 1))
            self.creators[0].backward(new_grad * self.grads)

        # If tensor was created by a division operation.
        if self.creator_op == "div":
            new_grad1 = self.grads / self.creators[1].values
            new_grad2 = -self.grads * self.creators[0].values / (self.creators[1].values ** 2)
            self.creators[0].backward(new_grad1)
            self.creators[1].backward(new_grad2)

        # If tensor was created by the relu operation.
        if self.creator_op == "relu":
            self.creators[0].backward(self.grads * (self.values > 0))

    def shape(self):
        return self.values.shape
    
    def mean(self):
        return self.__class__(np.mean(self.values), creators=[self], creator_op="mean")  
    
    
class Linear():
    def __init__(self, input_dim, output_dim, name = None):
        #weights and biases are individual tensors 
        self.weights = Tensor(np.random.randn(input_dim, output_dim)*0)
        self.biases = Tensor(np.random.randn(1,output_dim)*0)
        self.name = name

    def forward(self, input):
        self.input = input
        return input * self.weights + self.biases

    def backward(self, grad):
        self.weights.backward(self.input.T().values * grad)
        self.biases.backward(grad)
        return grad @ self.weights.values.T