# CLAUDE.md

Agent-facing guide for this repo. User-facing docs: [README.md](README.md). Dataset invariants: [dataset/MAINTAINERS_NOTES.md](dataset/MAINTAINERS_NOTES.md).

## What this repo is

Benchmark harness that evaluates LLMs on 350 Qiskit-translated Quantum Katas tasks across 26 categories. Dataset is JSONL; harness runs each task's generated code against a paired test function and scores pass/fail with optional multi-run aggregation and prompting ablations.

## Layout

- [benchmark/](benchmark/) — harness package (installed as `benchmark`)
  - [config.py](benchmark/config.py) — `ModelConfig`, `ProviderType`, `PromptStrategy`, `SYSTEM_PROMPTS`, JSON model loading
  - [models.py](benchmark/models.py) — provider clients (anthropic / openai / google / litellm / vllm / qiskit_assistant)
  - [runner.py](benchmark/runner.py) — `BenchmarkRunner`, ablation orchestration, `qk-benchmark` CLI (`main()`)
  - [evaluator.py](benchmark/evaluator.py) — code extraction + test execution with a timeout
  - [isolation.py](benchmark/isolation.py) — isolated grader: the candidate runs in its own process and never sees the test or decides the verdict. Don't move the candidate back into the test's process (see issue #6)
  - [reporter.py](benchmark/reporter.py) — `qk-compare` CLI, Wilson CIs, markdown reports
- [dataset/qiskit_quantumkatas.jsonl](dataset/qiskit_quantumkatas.jsonl) — 350 tasks, one per line
- [models.json](models.json) — model registry loaded automatically by the CLI
- [results/](results/) — per-model result JSON and per-model checkpoint subdirs (`checkpoint_<vendor>/...`)
- [paper/](paper/) — LaTeX sources and figures; not runtime
- [examples/run_benchmark.py](examples/run_benchmark.py) — Python API example

## Tooling

- Python **3.10+**, managed with **uv** (never pip into the system env). Run everything via `uv run …`.
- Lint/format: `ruff` (config in [pyproject.toml](pyproject.toml), line length 100).
- Tests: `pytest` (testpaths = `tests`, currently minimal).
- CLI entry points (from `pyproject.toml`): `qk-benchmark`, `qk-compare`.

## Common commands

```bash
uv sync                                              # install deps
uv run qk-benchmark --list-models                    # list models from models.json
uv run qk-benchmark --model claude-sonnet-4.6       # full 350-task run
uv run qk-benchmark --model X --task-ids BasicGates/1.1  # single task (fast iteration)
uv run qk-benchmark --model X --ablation --num-runs 3    # 7-config ablation x 3 runs
uv run qk-benchmark --all --parallel 4              # all configured models in parallel
uv run qk-compare                                    # compare existing results/
```

Dataset validation: skill `/validate-dataset` (runs every canonical solution against its own test; all 350 must pass).

## Dataset invariants — do not "fix" without reading the maintainers' notes

1. **Six tasks have trivial-looking solutions that are correct by construction** (identity oracles, classical CHSH, `|0⟩_L = |000⟩`, etc.). Listed in [dataset/MAINTAINERS_NOTES.md](dataset/MAINTAINERS_NOTES.md). Any stub-probe audit will flag them — leave them alone.
2. **Five prompt stubs end with typed returns** (`return False`, `return (x1, x2, x3)`, etc.) to signal return shape. Don't normalize to `pass`.
3. **Preserve unicode math notation** in prompts and docs: `|ψ⟩`, `α`, `β`, `⊗`, `√`. Never ASCII-transliterate. If you see mojibake or `\\u27e9` escapes, decode to the real glyph — that's the "fix."
4. **`entry_point` is not unique** across tasks (e.g. `bell_state` appears in multiple). Always key by `task_id`.
5. Task schema: `{task_id, prompt, canonical_solution, test, entry_point}`. Tests call `entry_point` against a freshly built `QuantumCircuit`.

## Harness conventions

- **Provider selection** is keyed by `ProviderType` enum in [config.py](benchmark/config.py). Adding a provider means: enum entry + client in [models.py](benchmark/models.py) + env-var mapping in `ModelConfig.__post_init__`.
- **OpenAI-compatible base URLs must end in `/v1`** (vLLM, LiteLLM, Azure-compatible endpoints).
- **Custom `headers` in model config**: values are env-var *names*, resolved at request time (see README "Custom Headers").
- **Aggregation** (`--aggregate`): `majority` (default, >50%), `any` (pass@k), `all` (strict).
- **Temperatures** per model live in `models.json`; some reasoning models require `temperature=1.0` — don't silently normalize.
- **Parallel ablation**: each model runs its full 7-config ablation in its own worker process. Don't assume shared state.
- **Results are append-safe via checkpoints** — `checkpoint_<model>.json` / `checkpoint_<vendor>/` enables resume. Delete to force rerun; otherwise completed tasks are skipped.
- **Evaluator sandboxing** uses `multiprocessing` with captured stdout — don't replace with `exec()` in-process, you'll leak state between tasks.

## API keys / env

Loaded from `.env` via `python-dotenv` at CLI startup. Variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `VLLM_API_KEY`, `LITELLM_API_KEY`, `QISKIT_ASSISTANT_TOKEN`. Never commit `.env` or API keys; never add them to `models.json`.

## When editing

- Prefer small, surgical edits. Don't refactor provider code broadly — providers are intentionally parallel/duplicative for isolation.
- When touching the dataset JSONL, validate afterward (`/validate-dataset`) before committing; a broken canonical solution silently poisons the benchmark.
- Result JSONs are large and numerous — don't pretty-print or mass-rewrite them.
- The `paper/` LaTeX is independent; code changes don't require paper updates.

## License

CC-BY-NC-SA-4.0. Derived from Microsoft's QuantumKatas.
