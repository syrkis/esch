# %% paper.py
#     here we generate visuals for showcasing esch.
# by: Noah Syrkis

# Imports
from functools import partial, reduce

import jax.numpy as jnp
import numpy as np
from chex import dataclass
from einops import rearrange
from jax import lax, nn, random, tree, value_and_grad
from jaxtyping import Array, PyTree


# %% Constants
stride = (2, 2)
kernel_dims = [3, 3]
batch_size = 100
lr = 0.1
initializer = nn.initializers.he_normal()


@dataclass
class Params:
    enconv: PyTree[Array]
    deconv: PyTree[Array]
    encode: PyTree[Array]
    decode: PyTree[Array]


# %% load mnist
# ds = tfds.load("mnist", split="train")
# data = rearrange(jnp.array([x["image"] for x in tfds.as_numpy(ds)]), "(s b) h w c -> s b c h w", b=batch_size) / 255.0
# y = rearrange(jnp.array([x["label"] for x in tfds.as_numpy(ds)]), "(s b) -> s b", b=batch_size)


# %% Apply functions
def conv_fn(x, kernel):
    kernel = rearrange(kernel, "i o ... -> o i ...")
    return lax.conv(x, kernel, window_strides=stride, padding="SAME")


def deconv_fn(x, kernel):
    kernel = rearrange(kernel, "i o ... -> o i ...")
    return lax.conv_transpose(x, kernel, strides=stride, padding="SAME", dimension_numbers=("NCHW", "OIHW", "NCHW"))


def step_fn(fn, x, w):
    return nn.tanh(fn(x, w))


def encode_fn(x, params: Params):
    x = reduce(partial(step_fn, conv_fn), params.enconv, x)
    S, C, H, W = x.shape  # dimensions for later
    x = rearrange(x, "s c h w -> s (c h w)")
    x = reduce(partial(step_fn, jnp.matmul), params.encode, x)
    return x


def decode_fn(x, params: Params):
    x = reduce(partial(step_fn, jnp.matmul), params.decode, x)
    dim = np.sqrt(x.shape[1] / params.deconv[0].shape[0]).astype(np.int8)
    x = rearrange(x, "s (c h w) -> s c h w", s=x.shape[0], c=params.deconv[0].shape[0], h=dim, w=dim)
    x = reduce(partial(step_fn, deconv_fn), params.deconv[:-1], x)
    return nn.sigmoid(deconv_fn(x, params.deconv[-1]))


def apply_fn(params: Params, x):
    x = encode_fn(x, params)
    x = decode_fn(x, params)
    return x


# %% Init
def init_fn(rng, cnn_dims, mlp_dims):
    def aux(rng, dims, k=[]):
        rngs = random.split(rng, len(dims) - 1)
        shapes = [[i, o] + k for i, o in zip(dims[:-1], dims[1:])]
        return list(map(lambda k, s: initializer(k, s), rngs, shapes))

    rngs = random.split(rng, 4)
    params = Params(
        enconv=aux(rngs[0], cnn_dims, kernel_dims),
        encode=aux(rngs[1], mlp_dims),
        deconv=aux(rngs[2], cnn_dims[::-1], kernel_dims),
        decode=aux(rngs[3], mlp_dims[::-1]),
    )
    return params


# %% Training
@value_and_grad
def grad_fn(params, x):
    x_hat = apply_fn(params, x)
    return jnp.mean((x_hat - x) ** 2)


def update_fn(carry, x):
    loss, grads = grad_fn(carry, x)
    params = tree.map(lambda p, g: p - lr * g, carry, grads)
    return params, loss


def train_fn(params, data, epochs, scope_fn=lambda *_: None):
    # @scan_tqdm(epochs)
    def epoch_fn(params, epoch):
        params, loss = lax.scan(update_fn, params, data)
        scope = scope_fn(params, data[0])
        return params, (scope, loss)

    params, (scope, loss) = lax.scan(epoch_fn, params, jnp.arange(epochs))
    return params, (scope, loss)


# %%
cnn_dims, mlp_dims = [1, 16, 32], [1568, 64]
rng = random.PRNGKey(0)
params = init_fn(rng, cnn_dims, mlp_dims)
# params, (scope, loss) = train_fn(params, data, epochs=20, scope_fn=apply_fn)

# %%
# plt.plot(loss.flatten())
# out = rearrange(scope.squeeze(), "t b h w -> b t h w")
# esch.tile(out[:3], animated=True, path="out.svg", fps=1)
