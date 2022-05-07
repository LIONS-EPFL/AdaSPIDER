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


###PARAMS
batch_size = 1
layer_sizes = [28 * 28, 512, 512, 10]
step_size = 1
num_classes = 10
T = 3
algorithm = AdaGrad
optimizer = AdaGrad(eta=0.01, epsilon=0.01)
######

mnist_dataset = FashionMNIST("/tmp/mnist/", download=True, transform=FlattenAndCast())
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)
train_images = np.array(mnist_dataset.train_data).reshape(
    len(mnist_dataset.train_data), -1
)
train_labels = one_hot(np.array(mnist_dataset.train_labels), num_classes)
mnist_dataset_test = FashionMNIST("/tmp/mnist/", download=True, train=False)
test_images = jnp.array(
    mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1),
    dtype=jnp.float32,
)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), num_classes)


params = create_params(layer_sizes)
state = optimizer.create_state(params)
loss_lst = []
for epoch in range(T):
    state = algorithm.on_epoch_state_update(params, state, (train_images, train_labels))
    for x, y in training_generator:
        y = one_hot(y, num_classes)
        params, state = algorithm.update(params, state, (x,y))
        batch_loss = loss(params, x, y)
        loss_lst.append(batch_loss)
    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)
    print("#### Epoch {}".format(epoch))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))

np.savetxt("losses-" + optimizer.__str__()+".csv", np.asarray(loss_lst))