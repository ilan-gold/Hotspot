# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Changed

- Migrated to the [scverse cookiecutter template](https://github.com/scverse/cookiecutter-scverse) (hatch, ruff, pre-commit, MyST docs)
- Require Python ≥ 3.12
- Fully type-annotated, checked with `mypy --strict`, and ship a `py.typed` marker

### Fixed

- `Hotspot.create_modules` failing with `ValueError: underlying array is read-only` on pandas ≥ 3
