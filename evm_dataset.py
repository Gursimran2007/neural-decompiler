"""
DATA ENGINE + objective SCORING for the EVM-subset decompiler.

Same shape as the toy `dataset.py`, but the input side is now REAL EVM bytecode
tokens instead of toy bytecode. The target side is unchanged: the readable
arithmetic source. We reuse the toy's Vocab and the source language verbatim.

Two oracles, both grounded in 256-bit EVM arithmetic:
  - functional_equivalent : compare prediction vs the (hidden) original source
  - verified_equivalent   : compare prediction vs the BYTECODE itself, by
                            re-executing it — the real-world oracle that needs no
                            source code, only the on-chain bytecode.
"""

import random

import evm
import lang
from dataset import Vocab          # reuse the toy vocab verbatim

MOD = evm.MOD


# --------------------------------------------------------------------------- #
# Pair generation
# --------------------------------------------------------------------------- #
def generate_pair(rng: random.Random, max_depth: int, obfuscate: bool = False):
    """Return (evm_tokens, source_tokens, ast). Source/ast are always the CLEAN
    program; with obfuscate=True the BYTECODE carries injected dead code."""
    ast = lang.random_ast(rng, max_depth)
    source = lang.render(ast)
    code = evm.compile_ast(ast)
    if obfuscate:
        code = evm.obfuscate(code, rng)
    return code, lang.tokenize_source(source), ast


def generate_dataset(n: int, max_depth: int = 2, seed: int = 0,
                     dedup: bool = True, obfuscate: bool = False):
    rng = random.Random(seed)
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n and attempts < n * 50:
        attempts += 1
        code, src, ast = generate_pair(rng, max_depth, obfuscate)
        key = " ".join(code)
        if dedup and key in seen:
            continue
        seen.add(key)
        pairs.append((code, src, ast))
    return pairs


def build_vocabs(pairs):
    src_tokens, tgt_tokens = [], []
    for code, src, _ in pairs:
        src_tokens.extend(code)
        tgt_tokens.extend(src)
    return Vocab(src_tokens), Vocab(tgt_tokens)


# --------------------------------------------------------------------------- #
# Objective scoring (all arithmetic mod 2**256, to match the EVM exactly)
# --------------------------------------------------------------------------- #
def functional_equivalent(original_ast, predicted_source: str,
                          rng: random.Random, trials: int = 30) -> bool:
    """True if prediction computes the same value as the original SOURCE across
    random inputs, under EVM (mod 2**256) arithmetic."""
    pred_ast = lang.parse(predicted_source)
    if pred_ast is None:
        return False
    for _ in range(trials):
        env = {v: rng.randint(-5, 5) for v in lang.VARS}
        try:
            if lang.evaluate(original_ast, env) % MOD != lang.evaluate(pred_ast, env) % MOD:
                return False
        except Exception:
            return False
    return True


def verified_equivalent(evm_tokens, predicted_source: str,
                        rng: random.Random, trials: int = 40) -> bool:
    """The REAL-WORLD oracle: check a guess WITHOUT the source, by RE-EXECUTING
    the actual EVM bytecode. If predicted source and bytecode agree on the same
    random inputs (mod 2**256), the guess is provably a correct decompilation of
    this bytecode — exactly how you'd verify against an on-chain contract."""
    pred_ast = lang.parse(predicted_source)
    if pred_ast is None:
        return False
    for _ in range(trials):
        env = {v: rng.randint(-5, 5) for v in lang.VARS}
        try:
            if evm.run(evm_tokens, env) != lang.evaluate(pred_ast, env) % MOD:
                return False
        except Exception:
            return False
    return True
