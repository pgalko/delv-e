#!/usr/bin/env python3
"""
clear_embeddings.py — strip embeddings from a state.json so --continue
will re-embed all winning nodes via the backfill mechanism.

Used when the embedding source changes (e.g. finding_summary →
tested_estimand) and you want the geometry recomputed without re-running
the whole exploration loop.

Usage:
    python clear_embeddings.py output/state.json

Writes a .bak alongside the original before modifying.
"""
import json
import shutil
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: python clear_embeddings.py <path/to/state.json>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    with path.open() as f:
        state = json.load(f)

    tree = state.get('explorer', {}).get('insight_tree', {})
    cleared = 0
    for node in tree.values():
        if node.get('embedding') is not None:
            node['embedding'] = None
            node['embedding_model'] = None
            cleared += 1

    backup = path.with_suffix(path.suffix + '.bak')
    shutil.copy2(path, backup)

    with path.open('w') as f:
        json.dump(state, f, indent=2)

    print(f"Cleared embeddings on {cleared} nodes.")
    print(f"Backup: {backup}")
    print(f"Run with --continue --iterations 1 to trigger backfill.")


if __name__ == '__main__':
    main()