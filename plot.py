import numpy as np
from func import *
from optim import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3D
from scipy.signal import find_peaks
import argparse

# create the argument parser
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--f', type=str, default='f', help='Description of x')
parser.add_argument('--x', type=float, default=1, help='Description of x')
parser.add_argument('--y', type=float, default=1.5, help='Description of y')
parser.add_argument('--alpha', type=float, default=1, help='Description of alpha')
parser.add_argument('--beta', type=float, default=0.9, help='Description of beta')
parser.add_argument('--top_down', type=bool, default=False, help='Description of top_down')
parser.add_argument('--learning_rate', type=float, default=0.03, help='Description of learning_rate')
parser.add_argument('--num_epochs', type=int, default=100, help='Description of num_epochs')

# parse the arguments
args = vars(parser.parse_args())

# set the variables based on the arguments
x = args['x']
y = args['y']
alpha = args['alpha']
beta = args['beta']
top_down = args['top_down']
learning_rate = args['learning_rate']
num_epochs = args['num_epochs']
func = args['f']

functions = { 'f': f,
              'h': h,
              'g': g
            }

f = functions[func]()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
X = np.linspace(-3, 3, 100)
Y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(X, Y)
Z = f.forward(X, Y)

if top_down:
    ax.contour(X, Y, Z, levels=15, offset=-2, cmap='rainbow', zorder=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1,1,0.2])
    ax.set_axis_off()
    ax.view_init(elev=90, azim=0)
    if alpha > 0.3: alpha = 0.3
    
surf = ax.plot_surface(X, Y, Z, cmap=cm.rainbow, alpha=alpha, zorder=-1)

z = f.forward(x, y)
optimizer_paths = {
    "sgd": [(x, y, z)],
    "momentum": [(x, y, z)],
    "rmsprop": [(x, y, z)]
}

sgd_weights = [x, y]
mom_weights = [x, y]
rms_weights = [x, y]

# Define the hyperparameters

# Instantiate the optimizers
sgd_optimizer = SGD(lr=learning_rate)
momentum_optimizer = Momentum(beta=beta, lr=learning_rate)
rmsprop_optimizer = RMSprop(beta=beta, lr=learning_rate)

for i in range(num_epochs):
    
    # Update the parameters using SGD
    gradients = f.grad(sgd_weights[0], sgd_weights[1])
    sgd_weights = sgd_optimizer.step(sgd_weights, gradients)
    sgd_x, sgd_y = sgd_weights
    sgd_z = f.forward(sgd_x, sgd_y)
    optimizer_paths["sgd"].append((sgd_x, sgd_y, sgd_z))
    
    # Update the parameters using Momentum
    gradients = f.grad(mom_weights[0], mom_weights[1])
    mom_weights = momentum_optimizer.step(mom_weights, gradients)
    mom_x, mom_y = mom_weights
    mom_z = f.forward(mom_x, mom_y)
    optimizer_paths["momentum"].append((mom_x, mom_y, mom_z))
    
    # Update the parameters using RMSprop
    gradients = f.grad(rms_weights[0], rms_weights[1])
    rms_weights = rmsprop_optimizer.step(rms_weights, gradients)
    rms_x, rms_y = rms_weights
    rms_z = f.forward(rms_x, rms_y)
    optimizer_paths["rmsprop"].append((rms_x, rms_y, rms_z))
    

colors = ['r', 'g', 'b', 'p']
styles = ['--', '-.', ':', '-']
labels = ['SGD', 'Momentum', 'RMSprop']

for i, optim in enumerate(optimizer_paths):
    optimizer_path = np.array(optimizer_paths[optim])
    line = Line3D(optimizer_path[:, 0], optimizer_path[:, 1], optimizer_path[:, 2], color=colors[i], linestyle=styles[i], linewidth=2, zorder=10, label=labels[i]) # Add label argument
    ax.add_line(line)

min_indices = np.argwhere(Z == np.min(Z))
min_indices = [tuple(index) for index in min_indices]

for idx in min_indices:
    ax.scatter(X[idx], Y[idx], Z[idx], s=200, c='r', marker='x', zorder=5)

ax.legend()
plt.show()
