"""
A tiny reverse-mode AUTOMATIC DIFFERENTIATION engine — our own mini-PyTorch.

WHY THIS EXISTS
Training a network means computing the gradient of the loss w.r.t. every weight.
For our MLP earlier we did that calculus by hand. For a seq2seq with attention,
hand-derivation is huge and bug-prone. Instead we build autograd ONCE: every
operation records how to push gradients backward through itself, and a single
.backward() walks the whole computation graph applying the chain rule.

HOW IT WORKS (the key idea)
Each Tensor stores its value and a function `_backward` that knows how to send
gradient to its parents. When you do c = a + b, c remembers a and b. Calling
loss.backward() topologically orders the graph and runs every node's _backward
from output to inputs, accumulating each Tensor's `.grad`. That's it — the same
mechanism PyTorch/TensorFlow use, in ~150 lines.

We verify it against NUMERICAL gradients (finite differences) in test_autograd.py,
so we KNOW it's correct before building a model on it.
"""

import numpy as np


class Tensor:
    def __init__(self, data, _parents=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._parents = set(_parents)
        self._op = _op

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _as_tensor(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    # -- the trick for broadcasting: when a gradient was broadcast during the
    #    forward pass, sum it back down to the parent's original shape ---------
    @staticmethod
    def _unbroadcast(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    # -- elementwise ops -----------------------------------------------------
    def __add__(self, other):
        other = self._as_tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._as_tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = self._as_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.swapaxes(-1, -2)
            other.grad += self.data.swapaxes(-1, -2) @ out.grad
        out._backward = _backward
        return out

    def __pow__(self, p):
        out = Tensor(self.data ** p, (self,), f"**{p}")

        def _backward():
            self.grad += (p * self.data ** (p - 1)) * out.grad
        out._backward = _backward
        return out

    # -- unary / reductions --------------------------------------------------
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-12), (self,), "log")

        def _backward():
            self.grad += (1.0 / (self.data + 1e-12)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    # -- indexing (needed to gather a row / a timestep) ----------------------
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), "getitem")

        def _backward():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)
            self.grad += g
        out._backward = _backward
        return out

    # -- transpose (2D) ------------------------------------------------------
    @property
    def T(self):
        out = Tensor(self.data.T, (self,), "T")

        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out

    # -- softmax over the last axis (stable) ---------------------------------
    def softmax(self):
        z = self.data - self.data.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        out = Tensor(p, (self,), "softmax")

        def _backward():
            # Jacobian-vector product for softmax, per row.
            dot = (out.grad * p).sum(axis=-1, keepdims=True)
            self.grad += p * (out.grad - dot)
        out._backward = _backward
        return out

    # -- convenience ---------------------------------------------------------
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (self._as_tensor(other) * -1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # -- the engine: run the whole graph backwards ---------------------------
    def backward(self):
        # 1. topological order of all nodes feeding into self
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._parents:
                    build(p)
                topo.append(v)
        build(self)

        # 2. seed the output gradient and propagate in reverse order
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


# A Parameter is just a Tensor we intend to train.
def parameter(shape, scale):
    return Tensor(np.random.randn(*shape) * scale)


def cat(tensors, axis=0):
    """Concatenate a list of Tensors along `axis` (with correct backward)."""
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(data, tuple(tensors), "cat")

    sizes = [t.data.shape[axis] for t in tensors]

    def _backward():
        start = 0
        for t, sz in zip(tensors, sizes):
            sl = [slice(None)] * out.grad.ndim
            sl[axis] = slice(start, start + sz)
            t.grad += out.grad[tuple(sl)]
            start += sz
    out._backward = _backward
    return out
