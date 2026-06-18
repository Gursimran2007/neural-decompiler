"""
BYTECODE OBFUSCATION — make the decompiler robust to identity / dead-code.

A real reverse-engineer never gets clean bytecode. Obfuscators inject operations
that change the instruction stream WITHOUT changing what the program computes:

    x        becomes   x + 0          (add-zero)
    x        becomes   x * 1          (multiply-one)
    x        becomes   ( x + 7 ) - 7  (add-then-subtract, a "split" constant)

These are SEMANTIC NO-OPS: identical value for every input, but the bytecode is
longer and the real structure is buried under junk. The decompiler must learn to
see THROUGH them and still recover the CLEAN original source.

How we do it: wrap random AST nodes in value-preserving identities, then compile
THAT obfuscated tree to bytecode. The training target stays the clean source, so
the model is forced to strip the obfuscation. Functional equivalence is preserved
by construction — every wrapper below is an exact arithmetic identity over the
integers, using only the +, -, * the language already has (no division, so no
rounding traps).
"""

import random


def _add_zero(node, rng):          # node + 0
    return ("op", "+", node, ("num", 0))


def _sub_zero(node, rng):          # node - 0
    return ("op", "-", node, ("num", 0))


def _mul_one(node, rng):           # node * 1
    return ("op", "*", node, ("num", 1))


def _plus_k_minus_k(node, rng):    # ( node + k ) - k   — a constant split across the tree
    k = rng.randint(1, 9)
    return ("op", "-", ("op", "+", node, ("num", k)), ("num", k))


IDENTITIES = [_add_zero, _sub_zero, _mul_one, _plus_k_minus_k]


def obfuscate_ast(ast, rng: random.Random, prob: float = 0.3):
    """
    Return a NEW AST that computes the same value as `ast` but has dead code
    injected. We recurse first (so children may be obfuscated too), then with
    probability `prob` wrap the resulting node in one random identity.
    """
    kind = ast[0]
    if kind in ("num", "var"):
        node = ast
    else:
        _, op, left, right = ast
        node = ("op", op,
                obfuscate_ast(left, rng, prob),
                obfuscate_ast(right, rng, prob))
    if rng.random() < prob:
        node = rng.choice(IDENTITIES)(node, rng)
    return node
