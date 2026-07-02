@AGENTS.md

## Claude Code

Use this file as a thin bridge to the shared Codex workflow. Keep durable implementation context in `AGENTS.md` and docs under `docs/`.

Use Claude for focused changes to this implementation workspace, then report changed files, checks run, and any unresolved issues back to Codex for review.

## Environment notes

No committed venv; set one up before running tests:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,app,quantum]'
pytest -q && ruff check .
```

PennyLane's public API has moved under active development (e.g. `qml.specs()`
became dict-like rather than attribute-access, `qml.set_shots` became a
positional `(qnode, shots)` transform requiring `functools.partial` when used
as a bare decorator). `pyproject.toml` pins `pennylane>=0.40,<0.43`, last
verified against 0.42.3. If bumping this range, rerun `pytest -q
tests/test_pennylane.py` first — quantum tests are the ones that break.
