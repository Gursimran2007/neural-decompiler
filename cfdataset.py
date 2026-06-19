"""
DATA ENGINE + objective SCORING for the CONTROL-FLOW decompiler.

Same shape as evm_dataset.py, but the programs now contain comparisons and
if/else, and the input bytecode contains REAL control flow (JUMPI/JUMP/JUMPDEST
with resolved addresses). The verifier still grounds everything by RE-EXECUTING
the actual bytecode (now with a program counter that follows the jumps) — so a
"verified" answer is provably a correct decompilation of that bytecode, no
source needed. That re-execution oracle is the moat: the model may only commit
to answers it can prove, so it never hallucinates control flow.
"""

import random

import cfevm
import cflang
from dataset import Vocab          # reuse the seq2seq vocab verbatim

MOD = cfevm.MOD


# --------------------------------------------------------------------------- #
# Pair generation
# --------------------------------------------------------------------------- #
def generate_pair(rng: random.Random, max_depth: int):
    """Return (evm_tokens, source_tokens, ast) for one random control-flow program."""
    ast = cflang.random_program(rng, max_depth)
    source = cflang.render(ast)
    code = cfevm.compile_ast(ast)
    return code, cflang.tokenize_source(source), ast


def generate_dataset(n: int, max_depth: int = 2, seed: int = 0, dedup: bool = True):
    rng = random.Random(seed)
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n and attempts < n * 80:
        attempts += 1
        code, src, ast = generate_pair(rng, max_depth)
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
# Objective scoring (all arithmetic mod 2**256, signed comparisons)
# --------------------------------------------------------------------------- #
def functional_equivalent(original_ast, predicted_source: str,
                          rng: random.Random, trials: int = 30) -> bool:
    """True if prediction computes the same value as the original SOURCE across
    random inputs (EVM mod 2**256). Exercises both branches of every if."""
    pred_ast = cflang.parse(predicted_source)
    if pred_ast is None:
        return False
    for _ in range(trials):
        env = {v: rng.randint(-5, 5) for v in cflang.VARS}
        try:
            if cflang.evaluate(original_ast, env) % MOD != cflang.evaluate(pred_ast, env) % MOD:
                return False
        except Exception:
            return False
    return True


def verified_equivalent(evm_tokens, predicted_source: str,
                        rng: random.Random, trials: int = 40) -> bool:
    """The REAL-WORLD oracle: verify a guess WITHOUT the source, by RE-EXECUTING
    the actual bytecode (following its jumps). If predicted source and bytecode
    agree on the same random inputs, the guess is provably correct for this
    bytecode — exactly how you'd check against an on-chain contract."""
    pred_ast = cflang.parse(predicted_source)
    if pred_ast is None:
        return False
    for _ in range(trials):
        env = {v: rng.randint(-5, 5) for v in cflang.VARS}
        try:
            if cfevm.run(evm_tokens, env) != cflang.evaluate(pred_ast, env) % MOD:
                return False
        except Exception:
            return False
    return True
