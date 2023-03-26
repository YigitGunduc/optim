class SGD():
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def step(self, params, grads):
        for i in range(len(params)):
            params[i] = params[i] - (self.lr * grads[i])
    
        return params
    
class Momentum():
    def __init__(self, beta=0.9, lr=0.01):
        self.beta = beta
        self.lr = lr
        self.acc = 0
    
    def step(self, params, grads):
        for i in range(len(params)):
            self.acc = self.beta * self.acc + (1 - self.beta) * grads[i]
            params[i] = params[i] - (self.lr * self.acc)

        return params
    
class RMSprop():
    def __init__(self, beta=0.9, lr=0.01):
        self.beta = beta
        self.lr = lr
        self.eps = 1e-8
        self.acc = 0

    def step(self, params, grads):
        for i in range(len(params)):
            self.acc = self.beta * self.acc + (1 - self.beta) * grads[i]**2
            params[i] = params[i] - (self.lr * grads[i] / ((self.acc)**(0.5) + self.eps))

        return params
