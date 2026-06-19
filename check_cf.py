"""
Verify the CONTROL-FLOW VM before any ML — the project's bottom-up discipline.

We cross-check two independent implementations of "what a program means":
  1. cflang.evaluate  — interpret the AST directly (the source semantics)
  2. cfevm.run        — compile the AST to bytecode WITH JUMPS and execute it on
                        the program-counter machine (the bytecode semantics)

If the data engine is sound, these must agree (mod 2**256) on every program and
every input — including taking the correct branch of every `if`. We also check
that bytecode round-trips through real hex. Zero mismatches is the gate that
lets us trust everything trained on top.
"""

import random

import cfevm
import cflang

MOD = cfevm.MOD


def main():
    rng = random.Random(0)
    n_programs, n_inputs = 4000, 6
    mism = fails = both_branches = 0
    total_len = clen = 0

    for _ in range(n_programs):
        ast = cflang.random_program(rng, max_depth=2)
        code = cfevm.compile_ast(ast)
        # round-trip through real bytecode hex
        hexed = cfevm.to_bytes(code)
        assert hexed.startswith("0x")

        saw_true = saw_false = False
        for _ in range(n_inputs):
            env = {v: rng.randint(-9, 9) for v in cflang.VARS}
            try:
                want = cflang.evaluate(ast, env) % MOD
                got = cfevm.run(code, env)
            except Exception:
                fails += 1
                continue
            if want != got:
                mism += 1
                if mism <= 5:
                    print("MISMATCH:", cflang.render(ast))
                    print("   env:", env, "eval:", want, "run:", got)
            # track whether this program is a conditional that took both paths
            if ast[0] == "if":
                if cflang.evaluate(ast[1], env) != 0:
                    saw_true = True
                else:
                    saw_false = True
        if ast[0] == "if" and saw_true and saw_false:
            both_branches += 1
        total_len += 1
        clen += len(code)

    print(f"programs checked      : {n_programs} x {n_inputs} inputs")
    print(f"AST-eval == bytecode  : {'OK (0 mismatches)' if mism == 0 else f'{mism} MISMATCHES'}")
    print(f"execution failures    : {fails}")
    print(f"conditionals that took both branches in test: {both_branches}")
    print(f"avg bytecode length   : {clen / total_len:.1f} tokens")

    # one worked example
    rng2 = random.Random(7)
    while True:
        ast = cflang.random_program(rng2, 2)
        if ast[0] == "if":
            break
    code = cfevm.compile_ast(ast)
    print("\nExample control-flow program:")
    print("  source  :", cflang.render(ast))
    print("  bytecode:", " ".join(code))
    print("  hex     :", cfevm.to_bytes(code))


if __name__ == "__main__":
    main()
