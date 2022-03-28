# Adapted from https://github.com/karpathy/lecun1989-repro/blob/master/prepro.py
import argparse
from typing import Callable

import jax
import jax.numpy as jnp
from jax import value_and_grad
from torchvision import datasets
import optax
from flax import linen as nn
from flax.training import train_state
from flax.linen.activation import tanh

def get_datasets(n_tr, n_te):
    train_test = {}
    for split in {'train', 'test'}:
        data = datasets.MNIST('./data', train=split=='train', download=True)

        n = n_tr if split == 'train' else n_te
        key = jax.random.PRNGKey(0)
        rp = jax.random.permutation(key, len(data))[:n]

        X = jnp.full((n, 16, 16, 1), 0.0, dtype=jnp.float32)
        Y = jnp.full((n, 10), -1.0, dtype=jnp.float32)
        for i, ix in enumerate(rp):
            I, yint = data[int(ix)]
            xi = jnp.array(I, dtype=jnp.float32) / 127.5 - 1.0
            xi = jax.image.resize(xi, (16, 16), 'bilinear')
            X = X.at[i].set(jnp.expand_dims(xi, axis=2))
            Y = Y.at[i, yint].set(1.0)
        train_test[split] = (X, Y)
    return train_test

class Net(nn.Module):
    bias_init: Callable = nn.initializers.zeros
    kernel_init: Callable = nn.initializers.uniform()

    @nn.compact
    def __call__(self, x):
        # For weight initialization, Karpathy used numerator of 2.4 
        # which is very close to sqrt(6) = 2.449... used by he_uniform()
        # By default, weight-sharing forces bias-sharing and therefore
        # we add the bias separately.
        bias1 = self.param('bias1', self.bias_init, (8, 8, 12))
        bias2 = self.param('bias2', self.bias_init, (4, 4, 12))
        bias3 = self.param('bias3', self.bias_init, (30,))
        bias4 = self.param('bias4', nn.initializers.constant(-1.0), (10,))
        x = jnp.pad(x, [(0,0),(2,2),(2,2),(0,0)], constant_values=-1.0)
        x = nn.Conv(features=12, kernel_size=(5,5), strides=2, padding='VALID',
                    use_bias=False, kernel_init=self.kernel_init)(x)
        x = tanh(x + bias1)
        x = jnp.pad(x, [(0,0),(2,2),(2,2),(0,0)], constant_values=-1.0)
        x1, x2, x3 = (x[..., 0:8], x[..., 4:12], 
                      jnp.concatenate((x[..., 0:4], x[..., 8:12]), axis=-1))
        slice1 = nn.Conv(features=4, kernel_size=(5,5), strides=2, padding='VALID', 
                         use_bias=False, kernel_init=self.kernel_init)(x1)
        slice2 = nn.Conv(features=4, kernel_size=(5,5), strides=2, padding='VALID',
                         use_bias=False, kernel_init=self.kernel_init)(x2)
        slice3 = nn.Conv(features=4, kernel_size=(5,5), strides=2, padding='VALID',
                         use_bias=False, kernel_init=self.kernel_init)(x3)
        x = jnp.concatenate((slice1, slice2, slice3), axis=-1)
        x = tanh(x + bias2)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=30, use_bias=False)(x)
        x = tanh(x + bias3)
        x = nn.Dense(features=10, use_bias=False)(x)
        x = tanh(x + bias4)
        return x

def create_train_state(key, lr, X):
    model = Net()
    params = model.init(key, X)['params']
    sgd_opt = optax.sgd(lr)
    return train_state.TrainState.create(apply_fn=model.apply, 
                                         params=params, tx=sgd_opt)

@jax.jit
def train_step(state, X, Y):
    def loss_fn(params):
        Yhat = Net().apply({'params': params}, X)
        loss = jnp.mean((Yhat - Y)**2)
        err = jnp.mean(jnp.argmax(Y, -1) != jnp.argmax(Yhat, -1)).astype(float)
        return loss, err
    (_, Yhats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state

def train_one_epoch(state, X, Y):
    for step_num in range(X.shape[0]):
        x, y = jnp.expand_dims(X[step_num], 0), jnp.expand_dims(Y[step_num], 0)
        state = train_step(state, x, y)
    return state

def train(key, data, epochs, lr):
    Xtr, Ytr = data['train']
    Xte, Yte = data['test']
    train_state = create_train_state(key, lr, Xtr)
    for epoch in range(epochs):
        print(f"epoch {epoch+1}")
        train_state = train_one_epoch(train_state, Xtr, Ytr)
        for split in ['train', 'test']:
            eval_split(data, split, train_state.params)

@jax.jit
def eval_step(params, X, Y):
    Yhat = Net().apply({'params': params}, X)
    loss = jnp.mean((Yhat - Y)**2)
    err = jnp.mean(jnp.argmax(Y, -1) != jnp.argmax(Yhat, -1)).astype(float)
    return loss, err

def eval_split(data, split, params):
    X, Y = data[split]
    loss, err = eval_step(params, X, Y)
    print(f"eval: split {split:5s}. loss {loss:e}. "
          f"error {err*100:.2f}%. misses: {int(err*Y.shape[0])}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a 1989 LeCun ConvNet on digits")
    parser.add_argument('--learning-rate', '-l', type=float, default=0.03, help="SGD learning rate")
    parser.add_argument('--output-dir'   , '-o', type=str,   default='out/base', help="output directory for training logs")
    args = parser.parse_args()
    print(vars(args))
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    
    train(key, get_datasets(7291, 2007), 23, args.learning_rate)