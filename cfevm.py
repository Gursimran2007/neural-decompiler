"""
EVM-subset machine WITH REAL CONTROL FLOW — the milestone past arithmetic.

This extends evm.py from a straight-line stack machine to a genuine
program-counter machine that follows JUMP / JUMPI just like a real Ethereum
node. That is the whole point: an `if` in the source has NO `if` in the
bytecode — it compiles to a conditional jump over a block, with the absolute
byte address resolved by a two-pass assembler. Decompiling means recovering the
`if/else` structure back out of that jump soup.

What is REAL here (matches mainnet semantics):
  - Genuine opcode bytes incl. control flow: JUMP=0x56, JUMPI=0x57,
    JUMPDEST=0x5b, and comparisons SLT=0x12, SGT=0x13, EQ=0x14. PUSH2=0x61
    carries a 2-byte jump address, exactly how Solidity emits jump targets.
  - A program-counter interpreter: JUMP/JUMPI move execution to a byte offset
    that must land on a JUMPDEST (we validate it), 256-bit modular arithmetic,
    signed comparisons via two's-complement over 2**256.
  - Inputs via CALLDATALOAD at offsets 0x00/0x20/0x40/0x60 (Solidity arg layout).

The classic if/else compilation shape (cond on the stack, then):
    PUSH2 Ltrue ; JUMPI        ; if cond != 0 jump to the THEN block
    <else block>
    PUSH2 Lend  ; JUMP         ; skip the THEN block
  Ltrue: JUMPDEST
    <then block>
  Lend:  JUMPDEST
"""

MOD = 1 << 256
WORD = 32
VAR_OFFSETS = {"a": 0x00, "b": 0x20, "c": 0x40, "d": 0x60}

# Mnemonic -> opcode byte for the (now larger) subset we execute / emit.
SUBSET_TO_BYTE = {
    "STOP": 0x00, "ADD": 0x01, "MUL": 0x02, "SUB": 0x03,
    "SLT": 0x12, "SGT": 0x13, "EQ": 0x14,
    "CALLDATALOAD": 0x35, "POP": 0x50,
    "JUMP": 0x56, "JUMPI": 0x57, "JUMPDEST": 0x5B,
    "PUSH1": 0x60, "PUSH2": 0x61, "DUP1": 0x80, "SWAP1": 0x90,
}


def _op_size(op: str) -> int:
    return 2 if op == "PUSH1" else 3 if op == "PUSH2" else 1


# --------------------------------------------------------------------------- #
# Compile an AST to EVM-subset bytecode tokens, resolving jump addresses.
# --------------------------------------------------------------------------- #
def _emit(ast, out: list, ctr: list):
    kind = ast[0]
    if kind == "num":
        out.append({"op": "PUSH1", "arg": ast[1] & 0xFF})
        return
    if kind == "var":
        out.append({"op": "PUSH1", "arg": VAR_OFFSETS[ast[1]]})
        out.append({"op": "CALLDATALOAD"})
        return
    if kind == "if":
        l_true, l_end = ctr[0], ctr[0] + 1
        ctr[0] += 2
        _emit(ast[1], out, ctr)                          # condition
        out.append({"op": "PUSH2", "target": l_true})
        out.append({"op": "JUMPI"})
        _emit(ast[3], out, ctr)                           # else block
        out.append({"op": "PUSH2", "target": l_end})
        out.append({"op": "JUMP"})
        out.append({"op": "JUMPDEST", "label": l_true})
        _emit(ast[2], out, ctr)                            # then block
        out.append({"op": "JUMPDEST", "label": l_end})
        return
    _, op, left, right = ast
    _emit(left, out, ctr)
    _emit(right, out, ctr)
    if op == "+":
        out.append({"op": "ADD"})
    elif op == "*":
        out.append({"op": "MUL"})
    elif op == "-":
        out.append({"op": "SWAP1"}); out.append({"op": "SUB"})
    elif op == "<":
        out.append({"op": "SWAP1"}); out.append({"op": "SLT"})
    elif op == ">":
        out.append({"op": "SWAP1"}); out.append({"op": "SGT"})
    elif op == "==":
        out.append({"op": "EQ"})
    else:
        raise ValueError(f"unknown op {op!r}")


def compile_ast(ast) -> list[str]:
    """AST -> EVM-subset mnemonic tokens with REAL resolved jump addresses."""
    items: list = []
    _emit(ast, items, [0])
    # pass 1: assign a byte program-counter to every item; record JUMPDEST pcs
    pc = 0
    label_pc = {}
    for it in items:
        it["pc"] = pc
        if it["op"] == "JUMPDEST":
            label_pc[it["label"]] = pc
        pc += _op_size(it["op"])
    # pass 2: fill jump targets now that landing pads have addresses
    for it in items:
        if it["op"] == "PUSH2" and "target" in it:
            it["arg"] = label_pc[it["target"]]
    # flatten to tokens (operands as inline hex, like a real disassembly)
    tokens: list[str] = []
    for it in items:
        if it["op"] == "PUSH1":
            tokens += ["PUSH1", f"0x{it['arg']:02x}"]
        elif it["op"] == "PUSH2":
            tokens += ["PUSH2", f"0x{it['arg']:04x}"]
        else:
            tokens.append(it["op"])
    return tokens


# --------------------------------------------------------------------------- #
# Execute with a PROGRAM COUNTER that follows jumps (real EVM control flow).
# --------------------------------------------------------------------------- #
def _calldata(env: dict) -> bytes:
    buf = bytearray(WORD * len(VAR_OFFSETS))
    for v, off in VAR_OFFSETS.items():
        buf[off:off + WORD] = (env[v] % MOD).to_bytes(WORD, "big")
    return bytes(buf)


def _to_signed(x: int) -> int:
    return x - MOD if x >= (MOD >> 1) else x


def _pc_index(tokens: list[str]):
    """Map each instruction's byte program-counter -> its token index, and find
    the set of valid JUMPDEST byte addresses (jumps may only land there)."""
    pc_to_idx, jumpdests = {}, set()
    pc, k = 0, 0
    while k < len(tokens):
        pc_to_idx[pc] = k
        t = tokens[k]
        if t == "JUMPDEST":
            jumpdests.add(pc)
        pc += _op_size(t)
        k += 2 if t in ("PUSH1", "PUSH2") else 1
    return pc_to_idx, jumpdests


def run(tokens: list[str], env: dict, max_steps: int = 100000) -> int:
    """Execute EVM-subset bytecode with real JUMP/JUMPI control flow."""
    calldata = _calldata(env)
    pc_to_idx, jumpdests = _pc_index(tokens)
    stack: list[int] = []
    k = steps = 0
    while k < len(tokens) and steps < max_steps:
        steps += 1
        t = tokens[k]
        if t == "PUSH1" or t == "PUSH2":
            stack.append(int(tokens[k + 1], 16) % MOD)
            k += 2
            continue
        if t == "CALLDATALOAD":
            off = stack.pop()
            chunk = calldata[off:off + WORD].ljust(WORD, b"\x00")
            stack.append(int.from_bytes(chunk, "big"))
        elif t == "ADD":
            x, y = stack.pop(), stack.pop(); stack.append((x + y) % MOD)
        elif t == "MUL":
            x, y = stack.pop(), stack.pop(); stack.append((x * y) % MOD)
        elif t == "SUB":                                  # top - second
            x, y = stack.pop(), stack.pop(); stack.append((x - y) % MOD)
        elif t == "SLT":                                  # signed top < second
            a, b = stack.pop(), stack.pop()
            stack.append(1 if _to_signed(a) < _to_signed(b) else 0)
        elif t == "SGT":
            a, b = stack.pop(), stack.pop()
            stack.append(1 if _to_signed(a) > _to_signed(b) else 0)
        elif t == "EQ":
            a, b = stack.pop(), stack.pop(); stack.append(1 if a == b else 0)
        elif t == "DUP1":
            stack.append(stack[-1])
        elif t == "SWAP1":
            stack[-1], stack[-2] = stack[-2], stack[-1]
        elif t == "POP":
            stack.pop()
        elif t == "JUMP":
            dest = stack.pop()
            if dest not in jumpdests:
                raise ValueError(f"JUMP to non-JUMPDEST 0x{dest:x}")
            k = pc_to_idx[dest]
            continue
        elif t == "JUMPI":
            dest, cond = stack.pop(), stack.pop()
            if cond != 0:
                if dest not in jumpdests:
                    raise ValueError(f"JUMPI to non-JUMPDEST 0x{dest:x}")
                k = pc_to_idx[dest]
                continue
            k += 1
            continue
        elif t == "JUMPDEST":
            pass
        elif t == "STOP":
            break
        else:
            raise ValueError(f"unsupported opcode in run(): {t}")
        k += 1
    return stack[-1] % MOD


# --------------------------------------------------------------------------- #
# Real bytecode  <->  tokens
# --------------------------------------------------------------------------- #
def to_bytes(tokens: list[str]) -> str:
    """Assemble tokens into a real bytecode hex string a node would execute."""
    out = bytearray()
    k = 0
    while k < len(tokens):
        t = tokens[k]
        if t == "PUSH1":
            out.append(0x60)
            out.append(int(tokens[k + 1], 16) & 0xFF)
            k += 2
            continue
        if t == "PUSH2":
            out.append(0x61)
            out += (int(tokens[k + 1], 16) & 0xFFFF).to_bytes(2, "big")
            k += 2
            continue
        out.append(SUBSET_TO_BYTE[t])
        k += 1
    return "0x" + out.hex()
