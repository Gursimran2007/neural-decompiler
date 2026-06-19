"""
A real-EVM-SUBSET stack machine: the bridge from our toy to actual blockchains.

The Ethereum Virtual Machine (EVM) is the computer every Ethereum smart contract
runs on. Like our toy VM it is a STACK MACHINE, which is exactly why this project
ports over so naturally — the decompiler's whole skill is rebuilding tree
structure from a flat stack stream, and EVM bytecode is that same flat stack
stream, just with real-world opcodes.

What is REAL here (not toy):
  - Genuine EVM opcode bytes: ADD=0x01, MUL=0x02, SUB=0x03, PUSH1=0x60,
    DUP1=0x80, SWAP1=0x90, POP=0x50, CALLDATALOAD=0x35. `to_bytes` emits
    bytecode a real Ethereum node would execute identically.
  - 256-bit modular arithmetic. Every EVM value is an unsigned integer mod 2**256
    and every op wraps. We match that, so "run the bytecode" means the same thing
    here as on-chain (this is what keeps our verifier sound).
  - Inputs arrive via CALLDATALOAD — the same mechanism Solidity uses to read a
    function's arguments from the transaction calldata. Our four variables a,b,c,d
    live at calldata word offsets 0x00, 0x20, 0x40, 0x60.

What is a SUBSET (deliberately): we implement only the arithmetic + stack-shuffle
opcodes an expression needs. Real contracts also use memory, storage, jumps, and
hashing — `evm_fetch.py` measures exactly how big that gap is, honestly.
"""

MOD = 1 << 256                      # every EVM word is an integer mod 2**256
WORD = 32                           # 32 bytes per word

# Our four "function arguments", placed at these calldata offsets (like Solidity).
VAR_OFFSETS = {"a": 0x00, "b": 0x20, "c": 0x40, "d": 0x60}

# Mnemonic -> opcode byte, for the subset we actually execute / emit.
SUBSET_TO_BYTE = {
    "STOP": 0x00, "ADD": 0x01, "MUL": 0x02, "SUB": 0x03,
    "CALLDATALOAD": 0x35, "POP": 0x50, "PUSH1": 0x60,
    "DUP1": 0x80, "SWAP1": 0x90,
}


# --------------------------------------------------------------------------- #
# Compile an AST to EVM-subset bytecode tokens (post-order, like any compiler)
# --------------------------------------------------------------------------- #
def compile_ast(ast) -> list[str]:
    """AST -> list of EVM mnemonic tokens leaving the result on top of stack.

    - a constant n      ->  PUSH1 0x0n
    - a variable v      ->  PUSH1 <offset> CALLDATALOAD     (load the argument)
    - left + right      ->  <left> <right> ADD
    - left * right      ->  <left> <right> MUL
    - left - right      ->  <left> <right> SWAP1 SUB
        SUB computes top-minus-second. After pushing left then right, `right` is
        on top, so a SWAP1 first makes it compute left - right (not right - left).
        Real Solidity does the same operand juggling.
    """
    kind = ast[0]
    if kind == "num":
        return ["PUSH1", f"0x{ast[1]:02x}"]
    if kind == "var":
        return ["PUSH1", f"0x{VAR_OFFSETS[ast[1]]:02x}", "CALLDATALOAD"]
    _, op, left, right = ast
    body = compile_ast(left) + compile_ast(right)
    if op == "+":
        return body + ["ADD"]
    if op == "*":
        return body + ["MUL"]
    return body + ["SWAP1", "SUB"]


# --------------------------------------------------------------------------- #
# Execute EVM-subset bytecode on a stack machine (256-bit modular arithmetic)
# --------------------------------------------------------------------------- #
def _calldata(env: dict) -> bytes:
    """Lay out a,b,c,d as 32-byte big-endian words, exactly like tx calldata."""
    buf = bytearray(WORD * len(VAR_OFFSETS))
    for v, off in VAR_OFFSETS.items():
        word = (env[v] % MOD).to_bytes(WORD, "big")
        buf[off:off + WORD] = word
    return bytes(buf)


def run(tokens: list[str], env: dict) -> int:
    """Run EVM-subset mnemonic tokens; return the top of stack (mod 2**256)."""
    calldata = _calldata(env)
    stack: list[int] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "PUSH1":
            stack.append(int(tokens[i + 1], 16) % MOD)
            i += 2
            continue
        if t == "CALLDATALOAD":
            off = stack.pop()
            chunk = calldata[off:off + WORD].ljust(WORD, b"\x00")
            stack.append(int.from_bytes(chunk, "big"))
        elif t == "ADD":
            x, y = stack.pop(), stack.pop()
            stack.append((x + y) % MOD)
        elif t == "MUL":
            x, y = stack.pop(), stack.pop()
            stack.append((x * y) % MOD)
        elif t == "SUB":                      # top - second
            x, y = stack.pop(), stack.pop()
            stack.append((x - y) % MOD)
        elif t == "DUP1":
            stack.append(stack[-1])
        elif t == "SWAP1":
            stack[-1], stack[-2] = stack[-2], stack[-1]
        elif t == "POP":
            stack.pop()
        elif t == "STOP":
            break
        else:
            raise ValueError(f"unsupported opcode in run(): {t}")
        i += 1
    return stack[-1] % MOD


# --------------------------------------------------------------------------- #
# Real bytecode  <->  tokens  (so we can emit / ingest on-chain hex)
# --------------------------------------------------------------------------- #
def to_bytes(tokens: list[str]) -> str:
    """Assemble mnemonic tokens into a real bytecode hex string (e.g. '0x6005...')."""
    out = bytearray()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "PUSH1":
            out.append(0x60)
            out.append(int(tokens[i + 1], 16) & 0xFF)
            i += 2
            continue
        out.append(SUBSET_TO_BYTE[t])
        i += 1
    return "0x" + out.hex()


# Full opcode name table — used by from_bytes / evm_fetch to DISASSEMBLE real
# contracts (which use far more than our executable subset). Unknown bytes are
# reported as UNKNOWN_0xNN so the coverage report is honest.
FULL_NAMES = {
    0x00: "STOP", 0x01: "ADD", 0x02: "MUL", 0x03: "SUB", 0x04: "DIV",
    0x05: "SDIV", 0x06: "MOD", 0x07: "SMOD", 0x08: "ADDMOD", 0x09: "MULMOD",
    0x0A: "EXP", 0x0B: "SIGNEXTEND", 0x10: "LT", 0x11: "GT", 0x12: "SLT",
    0x13: "SGT", 0x14: "EQ", 0x15: "ISZERO", 0x16: "AND", 0x17: "OR",
    0x18: "XOR", 0x19: "NOT", 0x1A: "BYTE", 0x1B: "SHL", 0x1C: "SHR",
    0x1D: "SAR", 0x20: "KECCAK256", 0x30: "ADDRESS", 0x31: "BALANCE",
    0x32: "ORIGIN", 0x33: "CALLER", 0x34: "CALLVALUE", 0x35: "CALLDATALOAD",
    0x36: "CALLDATASIZE", 0x37: "CALLDATACOPY", 0x38: "CODESIZE",
    0x39: "CODECOPY", 0x3A: "GASPRICE", 0x3B: "EXTCODESIZE", 0x3D: "RETURNDATASIZE",
    0x3E: "RETURNDATACOPY", 0x40: "BLOCKHASH", 0x41: "COINBASE", 0x42: "TIMESTAMP",
    0x43: "NUMBER", 0x45: "GASLIMIT", 0x47: "SELFBALANCE", 0x48: "BASEFEE",
    0x50: "POP", 0x51: "MLOAD", 0x52: "MSTORE", 0x53: "MSTORE8", 0x54: "SLOAD",
    0x55: "SSTORE", 0x56: "JUMP", 0x57: "JUMPI", 0x58: "PC", 0x59: "MSIZE",
    0x5A: "GAS", 0x5B: "JUMPDEST", 0xF0: "CREATE", 0xF1: "CALL", 0xF2: "CALLCODE",
    0xF3: "RETURN", 0xF4: "DELEGATECALL", 0xF5: "CREATE2", 0xFA: "STATICCALL",
    0xFD: "REVERT", 0xFE: "INVALID", 0xFF: "SELFDESTRUCT",
}
for _n in range(1, 33):                       # PUSH1..PUSH32
    FULL_NAMES[0x5F + _n] = f"PUSH{_n}"
for _n in range(1, 17):                        # DUP1..DUP16, SWAP1..SWAP16
    FULL_NAMES[0x7F + _n] = f"DUP{_n}"
    FULL_NAMES[0x8F + _n] = f"SWAP{_n}"


def from_bytes(hexstr: str) -> list[str]:
    """Disassemble a real bytecode hex string into mnemonic tokens.

    Handles the WHOLE instruction set (not just our subset) so we can ingest and
    analyse on-chain contracts. PUSHn opcodes carry n inline operand bytes."""
    h = hexstr[2:] if hexstr.startswith("0x") else hexstr
    code = bytes.fromhex(h)
    out: list[str] = []
    i = 0
    while i < len(code):
        b = code[i]
        name = FULL_NAMES.get(b, f"UNKNOWN_0x{b:02x}")
        if 0x60 <= b <= 0x7F:                  # PUSHn: consume n operand bytes
            n = b - 0x5F
            operand = code[i + 1:i + 1 + n]
            out.append(name)
            out.append("0x" + operand.hex())
            i += 1 + n
        else:
            out.append(name)
            i += 1
    return out


# --------------------------------------------------------------------------- #
# Obfuscation with REAL EVM stack opcodes (the obfuscator's actual toolkit)
# --------------------------------------------------------------------------- #
# Each gadget is a value-preserving no-op; the comment is its minimum stack depth.
_GADGETS = [
    (["DUP1", "POP"], 1),                       # copy top, throw it away
    (["PUSH1", "0x00", "ADD"], 1),              # x + 0
    (["PUSH1", "0x01", "MUL"], 1),              # x * 1
    (["SWAP1", "SWAP1"], 2),                     # swap top two, twice
]


def _depth_delta(tok: str) -> int:
    if tok in ("PUSH1", "DUP1"):
        return 1
    if tok in ("ADD", "MUL", "SUB", "POP"):
        return -1
    return 0                                    # CALLDATALOAD, SWAP1, operands


def obfuscate(tokens: list[str], rng, prob: float = 0.25) -> list[str]:
    """Inject value-preserving dead code made of REAL stack opcodes.

    We simulate stack depth while walking the clean token stream, and at random
    points splice in a gadget whose depth precondition is met. The computed value
    is unchanged (verified by re-execution in check_evm.py) but the instruction
    stream is longer and the structure is buried — exactly what an EVM obfuscator
    does to frustrate analysts."""
    out: list[str] = []
    depth = 0
    for tok in tokens:
        if tok.startswith("0x"):                # operand of the preceding PUSH1
            out.append(tok)
            continue
        if rng.random() < prob:
            usable = [g for g, need in _GADGETS if depth >= need]
            if usable:
                out.extend(rng.choice(usable))
        out.append(tok)
        depth += _depth_delta(tok)
    return out
