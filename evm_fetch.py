"""
Pull REAL deployed contract bytecode from Ethereum mainnet and analyse it — an
honest reality-check on the gap between this toy and production decompilation.

We hit a public JSON-RPC node (no API key) with `eth_getCode`, disassemble the
returned bytecode with our full opcode table, and report how much of each real
contract falls inside the arithmetic+stack subset our model actually handles.

The point is intellectual honesty, not hype: our model decompiles clean
arithmetic expressions. Real contracts are mostly dispatch logic, storage,
memory and jumps. This script QUANTIFIES that gap instead of pretending it away —
and shows the obvious next milestone (handle control flow + storage).

Run: /opt/anaconda3/bin/python evm_fetch.py
"""

import json
import urllib.request
from collections import Counter

import evm

# Public RPC endpoints that serve eth_getCode without an API key. Tried in order.
ENDPOINTS = [
    "https://ethereum-rpc.publicnode.com",
    "https://eth.drpc.org",
    "https://cloudflare-eth.com",
]

# A handful of well-known mainnet contracts (public addresses).
CONTRACTS = {
    "WETH":              "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "DAI":               "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "USDC (proxy)":      "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "Uniswap V2 Router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
    "Uniswap V3 Factory":"0x1F98431c8aD98523631AE4a59f267346ea31F984",
}

# The opcodes our EVM-subset VM can actually execute / our model is trained on.
EXECUTABLE_SUBSET = {"STOP", "ADD", "MUL", "SUB", "CALLDATALOAD",
                     "POP", "PUSH1", "DUP1", "SWAP1"}


def fetch_code(address: str) -> str | None:
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_getCode",
                          "params": [address, "latest"]}).encode()
    for ep in ENDPOINTS:
        try:
            req = urllib.request.Request(
                ep, data=payload,
                headers={"Content-Type": "application/json",
                         "User-Agent": "Mozilla/5.0"})
            r = urllib.request.urlopen(req, timeout=20)
            code = json.load(r).get("result", "")
            if code and code != "0x":
                return code
        except Exception:
            continue
    return None


def analyse(name: str, code: str, agg: Counter):
    tokens = evm.from_bytes(code)
    opcodes = [t for t in tokens if not t.startswith("0x")]   # drop push operands
    n = len(opcodes)
    counts = Counter(opcodes)
    agg.update(counts)
    in_subset = sum(c for op, c in counts.items() if op in EXECUTABLE_SUBSET)
    nbytes = (len(code) - 2) // 2
    print(f"\n  {name}")
    print(f"    bytecode size      : {nbytes} bytes")
    print(f"    instructions       : {n}")
    print(f"    distinct opcodes    : {len(counts)}")
    print(f"    within our subset   : {in_subset}/{n}  ({100*in_subset/n:.1f}%)")
    top = ", ".join(f"{op}×{c}" for op, c in counts.most_common(6))
    print(f"    most common         : {top}")


def main():
    print("Fetching real mainnet contract bytecode (public RPC, no API key)...")
    agg: Counter = Counter()
    fetched = 0
    for name, addr in CONTRACTS.items():
        code = fetch_code(addr)
        if code is None:
            print(f"\n  {name}: could not fetch (network/endpoint issue)")
            continue
        analyse(name, code, agg)
        fetched += 1

    if fetched == 0:
        print("\nNo contracts fetched — check network access, then re-run.")
        return

    total = sum(agg.values())
    in_subset = sum(c for op, c in agg.items() if op in EXECUTABLE_SUBSET)
    print("\n" + "=" * 64)
    print("HONEST REALITY CHECK (aggregate over fetched contracts)")
    print("=" * 64)
    print(f"  total instructions          : {total}")
    print(f"  inside our executable subset : {in_subset}  "
          f"({100*in_subset/total:.1f}%)")
    print(f"  top opcodes in the wild     :")
    for op, c in agg.most_common(12):
        mark = " (handled)" if op in EXECUTABLE_SUBSET else ""
        print(f"      {op:14s} {c:6d}  ({100*c/total:4.1f}%){mark}")
    print("\n  Takeaway: real contracts are dominated by control flow (JUMP/"
          "JUMPI/\n  JUMPDEST), memory (MLOAD/MSTORE) and storage (SLOAD/SSTORE)"
          " — none of\n  which our arithmetic model handles yet. That is exactly"
          " the next\n  milestone: extend the VM + model to control flow and "
          "storage, keeping\n  the same verified-by-re-execution oracle.")


if __name__ == "__main__":
    main()
