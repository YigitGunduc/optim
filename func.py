import numpy as np

class f():
    def __init__(self):
        pass

    def forward(self, x, y):
        return np.sin(x) + np.sin(2*y)

    def grad(self, x, y):
        x_grad = np.cos(x)
        y_grad = 2*np.cos(2*y)
        return [x_grad, y_grad]

class g():
    def __init__(self):
        pass

    def forward(self, x, y):
        return x**2 + 2*y**2 - x*y - 4*x + 3*y

    def grad(self, x, y):
        x_grad = 2*x - y - 4
        y_grad = 4*y - x + 3
        return [x_grad, y_grad]

class h():
    def __init__(self):
        pass

    def forward(self, x, y):
        return (x**2)/5 + y**2 - np.cos(2*np.pi*x)/5 - np.cos(2*np.pi*y)/5

    def grad(self, x, y):
        grad_x = (2*x/5) + (2*np.pi*np.sin(2*np.pi*x))/5
        grad_y = 2*y + (2*np.pi*np.sin(2*np.pi*y))/5
        return [grad_x, grad_y]
