# Repository Guidelines

## Project Structure & Module Organization
- Tooling lives at the repo root: `pyproject.toml` and `uv.lock` pin Python 3.12+, dependency extras (`cu124`), and the `uv` workflow described in `README.md`.
- The `nanogaia/` package holds the code: `main.py` is a simple entry point, `frame_reader.py` implements buffered video access, `comma2k19_dataset.py` exposes a `torch.utils.data.Dataset`, and `prepare_dataset.py` downloads + inflates Comma2k19 chunks into `dataset/`.
- Keep generated chunk folders (e.g., `dataset/Chunk_1/.../video.hevc` and derived `raw_images/`) out of git; reference them via absolute paths or `Path` helpers.

## Build, Test, and Development Commands
- Install OS deps then sync Python packages: `sudo apt install ffmpeg` and `uv sync --extra cu124`.
- Format before pushing: `uv run black nanogaia`.
- Run the package: `uv run python -m nanogaia.main` for a smoke check, or call scripts directly (`uv run python nanogaia/prepare_dataset.py`) to fetch all 10 chunks.
- Validate a prepared chunk via `uv run python nanogaia/comma2k19_dataset.py`, which instantiates `CommaDataset` and prints velocity tensors.

## Coding Style & Naming Conventions
- Default to PEP 8/Black (4-space indentation, 88-char lines, double quotes unless escaping). Run Black locally instead of relying on CI.
- Modules, functions, and files use `snake_case`; dataset/model classes use `PascalCase`; constants (e.g., chunk counts) should be `UPPER_SNAKE_CASE`.
- Favor typed signatures (`FrameReader` already type-hints return values) and concise docstrings explaining non-obvious transforms, especially around OpenCV/ffmpeg handling.

## Testing Guidelines
- There is no test harness yet; add `pytest`-based suites under `tests/` mirroring package paths (`tests/test_frame_reader.py`, `tests/data/test_comma_dataset.py`).
- Mock heavy assets by stubbing short clips or fixture numpy arrays stored under `dataset/fixtures/`.
- Minimum coverage expectation: assert frame shapes/dtypes, deterministic ordering, and error handling for missing chunk files. Run with `uv run pytest` before opening a PR.

## Commit & Pull Request Guidelines
- Existing history shows short, imperative subject lines (e.g., “enable prepare raw image”); continue that style and scope each commit narrowly.
- Reference related issues in the body (`Refs #123`), and describe dataset prerequisites (chunk IDs, HF tokens) so reviewers can reproduce.
- PRs should include: summary of behavior change, verification commands/output, screenshots if visual, and a note on dataset size or secrets touched. Request review only after `black` and `pytest` succeed locally.
