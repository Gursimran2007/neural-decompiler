"""
Verify the autograd engine: compare its analytic gradients against NUMERICAL
gradients (finite differences). If they match to ~1e-6, the chain-rule plumbing
is correct and we can trust it to train a network.

    numerical df/dx  ~=  ( f(x + h) - f(x - h) ) / (2h)

Run: /opt/anaconda3/bin/python test_autograd.py
"""

import numpy as np

from autograd import Tensor


def numerical_grad(f, x, h=1e-6):
    """Finite-difference gradient of scalar f w.r.t. each entry of array x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + h; fp = f(x)
        x[idx] = old - h; fm = f(x)
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * h)
        it.iternext()
    return grad


def check(name, build_scalar, x0):
    """build_scalar(np_array) -> python float ; same graph via Tensor for autograd."""
    # numerical
    ng = numerical_grad(build_scalar, x0.copy())
    # analytic
    t = Tensor(x0.copy())
    out = build_scalar(t)          # build graph with Tensor ops
    out.backward()
    err = np.abs(ng - t.grad).max()
    status = "OK " if err < 1e-4 else "FAIL"
    print(f"  [{status}] {name:22s} max|Δ| = {err:.2e}")
    return err < 1e-4


def scalarize(t):
    """Reduce any Tensor/array to a scalar so we can call .backward()."""
    return t.sum() if isinstance(t, Tensor) else float(np.sum(t))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ok = True

    # f = sum( (x @ W) tanh, then *2 + 1 )  — exercises matmul, tanh, add, mul, sum
    W = rng.standard_normal((4, 3))

    def f_linear_tanh(x):
        xt = x if isinstance(x, Tensor) else x
        Wt = Tensor(W) if isinstance(x, Tensor) else W
        if isinstance(x, Tensor):
            return ((x @ Wt).tanh() * 2 + 1).sum()
        return float(np.sum(np.tanh(x @ W) * 2 + 1))

    ok &= check("matmul+tanh+add+mul", f_linear_tanh, rng.standard_normal((2, 4)))

    # sigmoid, relu, pow
    ok &= check("sigmoid", lambda x: x.sigmoid().sum() if isinstance(x, Tensor)
                else float((1/(1+np.exp(-x))).sum()), rng.standard_normal((3, 3)))
    ok &= check("relu", lambda x: x.relu().sum() if isinstance(x, Tensor)
                else float(np.maximum(0, x).sum()), rng.standard_normal((3, 3)))
    ok &= check("pow2", lambda x: (x ** 2).sum() if isinstance(x, Tensor)
                else float((x ** 2).sum()), rng.standard_normal((3, 3)))

    # softmax then log (the core of cross-entropy)
    def f_logsoftmax(x):
        if isinstance(x, Tensor):
            return x.softmax().log().sum()
        z = x - x.max(axis=-1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
        return float(np.log(p + 1e-12).sum())
    ok &= check("softmax+log", f_logsoftmax, rng.standard_normal((3, 5)))

    # getitem (gather rows) — used to pick embeddings / timesteps
    def f_getitem(x):
        if isinstance(x, Tensor):
            return (x[1] * 3).sum()
        return float((x[1] * 3).sum())
    ok &= check("getitem", f_getitem, rng.standard_normal((4, 3)))

    print("\nAll gradients correct." if ok else "\nSOME GRADIENTS WRONG — fix before training.")
