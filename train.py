from jax import grad, tree_map
import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)

from utils import compute_distance, compute_gradient_norm, create_params, one_hot
from model import accuracy, loss, net
from dataloader import MNIST, FashionMNIST, NumpyLoader, CIFAR10
from optimizers import SGD, AdaGrad, AdaSVRG, AdaSpiderBoost, AdaSpider, Adam, KatyushaXw, Spider, AdaSpiderDiag, SpiderBoost
import wandb
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--step_size", default=0.01, type=float)
parser.add_argument("--eta", default=0.01, type=float)
parser.add_argument("--epsilon", default=1e-4, type=float)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--optimizer", default="SGD", type=str)
parser.add_argument("--dataset", default="MNIST", type=str)
parser.add_argument("--n", default=60000, type=int)
parser.add_argument("--L", default=10, type=float)
parser.add_argument("--run_id", default=0, type=int)
parser.add_argument("--beta_1", default=0.9, type=float)
parser.add_argument("--beta_2", default=0.999, type=float)

args = parser.parse_args()

###PARAMS
batch_size = args.batch_size
step_size = args.step_size
eta = args.eta
epsilon = args.epsilon
T = args.epochs
optimizers = {
    "SGD": SGD,
    "AdaSpider": AdaSpider,
    "AdaGrad": AdaGrad,
    "AdaSpiderBoost": AdaSpiderBoost,
    "Spider": Spider,
    "KatyushaXw": KatyushaXw,
    "AdaSpiderDiag": AdaSpiderDiag,
    "AdaSVRG": AdaSVRG,
    "Adam": Adam,
    "SpiderBoost": SpiderBoost
}
optimizer_params = {
    "SGD": {"step_size": args.step_size},
    "AdaSpider": {"n": args.n, "eta": eta},
    "AdaGrad": {"eta": eta, "epsilon": epsilon},
    "AdaSpiderBoost": {"eta": eta, "n": args.n},
    "Spider": {"n_zero": args.n, "L": args.L, "epsilon": args.epsilon},
    "KatyushaXw": {"step_size": args.step_size},
    "AdaSpiderDiag": {"n": args.n, "eta": eta},
    "AdaSVRG": {"eta": args.eta},
    "Adam": {"eta": args.eta, "epsilon": args.epsilon, "beta_1": args.beta_1, "beta_2": args.beta_2},
    "SpiderBoost": {"L": args.L}
}
algorithm = optimizers[args.optimizer]
optimizer = algorithm(**optimizer_params[args.optimizer])
selected_dataset = {"MNIST": MNIST, "FashionMNIST": FashionMNIST, "CIFAR10": CIFAR10}
data = selected_dataset[args.dataset]
######

######### RANDOM SEEDS ################
seeds = [1701, 42, 3427642, 93422287, 74]
seed = seeds[args.run_id]
torch.manual_seed(seed)
np.random.seed(seed)
rng = jax.random.PRNGKey(seeds[args.run_id])
#######################################

layer_sizes = [28 * 28, 512, 512, 10]
num_classes = 10

logger = wandb.init(
    project="Neurips-AdaSpider",
    name=optimizer.__str__() + "-" + args.dataset + "-run-" + str(args.run_id),
    config={"batch_size": batch_size, "epochs": T, "layer_sizes": layer_sizes},
    tags=[args.dataset],
)
wandb.config.update(args)

if args.dataset in ['MNIST', 'FashionMNIST']:
    shape = (28, 28, 1)
else:
    shape = (32, 32, 3)

class ReshapeAndCast(object):
    def __call__(self, pic):
        return (jnp.array(pic, dtype=jnp.float64)).reshape(*shape)

dataset = data("/tmp/mnist/", download=True, transform=ReshapeAndCast())
training_generator = NumpyLoader(dataset, batch_size=batch_size, num_workers=0)
train_images = jnp.array(dataset.data.numpy(), dtype=jnp.float64).reshape(len(dataset.data), *shape)
train_labels = one_hot(np.array(dataset.targets), num_classes)
dataset_test = data("/tmp/mnist/", download=True, train=False)
test_images = jnp.array(
    dataset_test.data.numpy().reshape(len(dataset_test.data), *shape),
    dtype=jnp.float64,
)
test_labels = one_hot(np.array(dataset_test.targets), num_classes)


params, net_state = net.init(rng, np.random.randn(args.batch_size, *shape), is_training=True)
starting_params = tree_map(lambda x: x.copy(), params)
state = optimizer.create_state(params)

train_acc = accuracy(params, net_state, train_images, train_labels)
test_acc = accuracy(params, net_state, test_images, test_labels)
gradient_norm = compute_gradient_norm(params, net_state, train_images, train_labels)
distance_from_init = compute_distance(params, starting_params)
logger.log(
    {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "grad_norm": gradient_norm,
        "distance_from_init": distance_from_init,
        "epoch": 0,
    }
)

for epoch in range(1, T+1):
    params, net_state, state = algorithm.on_epoch_state_update(params, net_state, state, (train_images, train_labels))
    for (idx, (x, y)) in enumerate(training_generator):
        y = one_hot(y, num_classes)
        net_state, state = algorithm.on_step_state_update(params, net_state, state, (x, y))
        params, net_state, state = algorithm.update(params, net_state, state, (x, y))
        batch_loss = loss(params, net_state, x, y)
        if "step_size" in state:
            logger.log({"step_size": state["step_size"]}, commit=False)
        if "acc_v" in state:
            logger.log({"accumulated_norm": state["acc_v"]}, commit=False)
        logger.log({"loss": batch_loss})

    train_acc = accuracy(params, net_state, train_images, train_labels)
    test_acc = accuracy(params, net_state, test_images, test_labels)
    gradient_norm = compute_gradient_norm(params, net_state, train_images, train_labels)
    distance_from_init = compute_distance(params, starting_params)
    logger.log(
        {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "grad_norm": gradient_norm,
            "distance_from_init": distance_from_init,
            "epoch": epoch,
        }
    )
    print("#### Epoch {}".format(epoch))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
