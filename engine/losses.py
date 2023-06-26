
class MSE:
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return ((pred.values - target.values) ** 2).mean()

    def backward(self):
        return ((self.pred.values - self.target.values) * 2) * (self.pred.values.size ** (-1))