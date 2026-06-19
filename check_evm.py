"""
Sanity-check the EVM-subset VM BEFORE training anything on it.

Three things must hold, or nothing downstream is trustworthy:
  1. AST eval (mod 2**256) == evm.run(bytecode)     — our compiler is correct
  2. obfuscated bytecode computes the SAME value     — dead code is truly dead
  3. tokens -> real bytes -> tokens round-trips       — our assembler is faithful

Run: /opt/anaconda3/bin/python check_evm.py
"""

import random

import evm
import lang

MOD = evm.MOD


def check_consistency(n=3000, max_depth=3):
    print("=== Consistency: AST eval (mod 2^256) == EVM bytecode eval ===")
    rng = random.Random(7)
    bad = bad_obf = 0
    lens = []
    for _ in range(n):
        ast = lang.random_ast(rng, max_depth)
        clean = evm.compile_ast(ast)
        dirty = evm.obfuscate(clean, rng)
        lens.append((len(clean), len(dirty)))
        for _ in range(5):
            env = {v: rng.randint(-9, 9) for v in lang.VARS}
            expected = lang.evaluate(ast, env) % MOD
            if evm.run(clean, env) != expected:
                bad += 1
            if evm.run(dirty, env) != expected:
                bad_obf += 1
    cl = sum(c for c, _ in lens) / len(lens)
    dl = sum(d for _, d in lens) / len(lens)
    print(f"  checked {n} programs x5 inputs")
    print(f"  clean mismatches      : {bad}")
    print(f"  obfuscated mismatches : {bad_obf}")
    print(f"  avg tokens clean {cl:.1f} -> obfuscated {dl:.1f}  ({dl/cl:.2f}x)")


def check_roundtrip(n=3000, max_depth=3):
    print("\n=== Round-trip: tokens -> real bytecode -> tokens ===")
    rng = random.Random(11)
    bad = 0
    for _ in range(n):
        ast = lang.random_ast(rng, max_depth)
        toks = evm.compile_ast(ast)
        if evm.from_bytes(evm.to_bytes(toks)) != toks:
            bad += 1
    print(f"  checked {n} programs | round-trip failures: {bad}")


def show_example():
    print("\n=== Example: source -> EVM bytecode ===")
    ast = lang.random_ast(random.Random(3), 2)
    toks = evm.compile_ast(ast)
    print(f"  source   : {lang.render(ast)}")
    print(f"  EVM asm  : {' '.join(toks)}")
    print(f"  bytecode : {evm.to_bytes(toks)}")
    env = {v: i for i, v in enumerate(lang.VARS, start=1)}
    print(f"  run({env}) = {evm.run(toks, env)}  "
          f"(AST says {lang.evaluate(ast, env) % MOD})")


if __name__ == "__main__":
    check_consistency()
    check_roundtrip()
    show_example()
    print("\nEVM engine looks good." )
