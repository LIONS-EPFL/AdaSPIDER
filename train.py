import numpy as np
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from utils import create_params, one_hot, create_state
from model import accuracy, loss
from algorithms import GD_update, ada_Grad_update
from dataloader import MNIST, FashionMNIST, NumpyLoader, FlattenAndCast
from optimizers import SGD, AdaGrad, SpiderBoost, AdaSpider
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--step_size', default=0.01, type=float)
parser.add_argument('--eta', default=0.01, type=float)
parser.add_argument('--epsilon', default=1e-4, type=float)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--optimizer', default='SGD', type=str)

args = parser.parse_args()

###PARAMS
batch_size = args.batch_size
step_size = args.step_size
eta = args.eta
epsilon = args.epsilon
T = args.epochs
optimizers = {'SGD': SGD, 'AdaSpider': AdaSpider, 'AdaGrad': AdaGrad}
optimizer_params = {'SGD': (step_size), 'AdaSpider': (50000), 'AdaGrad': (eta, epsilon)}
algorithm = optimizers[args.optimizer]
optimizer = algorithm(optimizer_params[args.optimizer])
######

layer_sizes = [28 * 28, 512, 512, 10]
num_classes = 10

logger = wandb.init(project="AdaSpider", name=optimizer.__str__(),
           config={"batch_size": batch_size, "epochs": T, "layer_sizes": layer_sizes})
wandb.config.update(args)

dataset = FashionMNIST("/tmp/mnist/", download=True, transform=FlattenAndCast())
training_generator = NumpyLoader(dataset, batch_size=batch_size, num_workers=0)
train_images = np.array(dataset.data).reshape(
    len(dataset.data), -1
)
train_labels = one_hot(np.array(dataset.targets), num_classes)
dataset_test = FashionMNIST("/tmp/mnist/", download=True, train=False)
test_images = jnp.array(
    dataset_test.data.numpy().reshape(len(dataset_test.data), -1),
    dtype=jnp.float32,
)
test_labels = one_hot(np.array(dataset_test.targets), num_classes)


params = create_params(layer_sizes)
state = optimizer.create_state(params)
loss_lst = []
for epoch in range(T):
    state = algorithm.on_epoch_state_update(params, state, (train_images, train_labels))
    for x, y in training_generator:
        y = one_hot(y, num_classes)
        params, state = algorithm.update(params, state, (x,y))
        batch_loss = loss(params, x, y)
        logger.log({"loss": batch_loss})
    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)
    logger.log({"train_acc": train_acc})
    logger.log({"test_acc": test_acc})
    print("#### Epoch {}".format(epoch))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
