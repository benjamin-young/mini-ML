class SGD:
    def __init__(self, parameters, alpha=0.00001):
        self.parameters = parameters
        self.alpha = alpha

    def step(self):
        for param in self.parameters:
            
            #print("param grads: " + str(param.grads.shape))
            #print("param shape: " + str(param.values.shape))
            param.values -= param.grads * self.alpha