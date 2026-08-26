# Repository Guidelines

## Project Structure & Module Organization

This repository implements UR10e manipulation tasks with MuJoCo and SmolVLA. `sim/` contains block and mug environments, `collector/` owns teleoperation and LeRobot dataset creation, `evaluate/` runs closed-loop evaluation, and `cloud/` launches training jobs. Put utilities in `scripts/`, settings in `configs/`, environment definitions in `env/`, and MuJoCo resources and licenses in `assets/`. Tests live in `tests/` and mirror these domains. Generated datasets and runs belong in ignored `smolvla-data/` and `outputs/`.

## Build, Test, and Development Commands

- `conda env create -f env/environment-collector.yml` creates the collection and simulation environment.
- `python -m scripts.view_mug_scene --headless --steps 10 --scene-seed 7` performs a short deterministic scene check.
- `python -m pytest tests/` runs the complete regression suite. On Linux, use `MUJOCO_GL=egl python -m pytest tests/` for headless rendering.
- `python -m pytest tests/test_chunk_blend.py::ChunkBlendPolicyTest::test_wrap_angle` runs one focused test.
- `python -m cloud.train --dry-run` validates a training configuration without starting training.
- `python -m evaluate --help` shows the local evaluation entry point.

Run commands from the repository root so asset and configuration paths resolve consistently.

## Coding Style & Naming Conventions

Use Python 3.10+ conventions, four-space indentation, type hints, and `pathlib.Path`. Name modules, functions, and variables `snake_case`, classes `PascalCase`, and constants `UPPER_SNAKE_CASE`. Follow existing Google-style docstrings; new script comments and docstrings should be clear Chinese unless an external interface requires English. Prefer `python -m package.module` CLI entry points. No formatter is enforced, so preserve nearby style and group imports as standard library, third party, then local.

## Testing Guidelines

Tests use pytest while many test classes inherit from `unittest.TestCase`. Name files `test_<domain>.py` and methods `test_<behavior>`. Add regression coverage beside the affected domain. Use fixed `scene_seed` values for reproducibility, close MuJoCo renderers in teardown, and use temporary directories for generated files. Run focused tests first, then the full suite when simulation or shared contracts change.

## Commit & Pull Request Guidelines

Follow the history's Conventional Commit pattern, such as `refactor(collector): simplify task definitions` or `docs(evaluate): update checkpoint examples`. Keep commits coherent. Pull requests should explain the motivation and behavior change, list verification commands, link related issues or experiment notes, and call out configuration or dataset-contract changes. Include screenshots or short videos for rendering changes; never include credentials, checkpoints, datasets, or generated media.
