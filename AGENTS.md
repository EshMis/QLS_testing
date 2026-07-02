# AI Agent Working Notes for QLS_testing

Start from the project README and docs under `docs/`.

Codex, Claude, and any other capable AI agent should treat these notes as shared implementation context. Durable updates should land here, in `docs/`, in tests/README files, or in the shared Obsidian project/provenance pages so the next agent sees the current state.

Shared local workflow:

This repo's owner keeps a cross-project workspace ("Tools&Agent Usage") as a
sibling folder outside this repo, on the local machine only (not part of this
git history). Use it for:

- paper ingestion and notes
- local retrieval over docs/literature
- slide outlines and editable PPTX export
- CV/project milestone updates

If you're an agent running on the repo owner's machine, ask for that path;
don't assume a specific absolute path — it varies by machine/user.

## Dev Setup

No committed virtualenv. Set one up before running tests:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,app,quantum]'
pytest -q && ruff check .
```

PennyLane's public API has moved under active development (e.g. `qml.specs()`
became dict-like rather than attribute-access, `qml.set_shots` became a
positional `(qnode, shots)` transform requiring `functools.partial` when used
as a bare decorator — see the fixes in `qls_testing/hardware_path/readout.py`
and `qls_testing/quantum_pennylane/vqls.py`). `pyproject.toml` pins
`pennylane>=0.40,<0.43`. If bumping this range, rerun
`pytest -q tests/test_pennylane.py` first — quantum tests are the ones that
break.

## Behavioral Guidelines (Karpathy / Forrest Chang)

Source: https://github.com/forrestchang/andrej-karpathy-skills (popular, widely-adopted CLAUDE.md behavioral guidelines derived from Andrej Karpathy's observations on LLM coding pitfalls). Applies to any agent working in this repo.

1. **Think before coding** - don't assume, don't hide confusion, surface tradeoffs. State assumptions; ask if uncertain; present multiple interpretations rather than silently picking one.
2. **Simplicity first** - minimum code that solves the problem, nothing speculative. No unrequested abstractions, flexibility, or error handling for impossible scenarios.
3. **Surgical changes** - touch only what you must, clean up only your own mess. Do not refactor or "improve" unrelated code. Remove only the imports/variables your own change orphaned.
4. **Goal-driven execution** - define success criteria, loop until verified. Turn tasks into verifiable goals (e.g. "fix the bug" -> "write a failing test that reproduces it, then make it pass").
