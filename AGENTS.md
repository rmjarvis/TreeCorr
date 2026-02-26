# TreeCorr Codex Notes

## Python Environment
Use the conda environment at:
- `/Users/Mike/miniforge3/envs/py3.12`

Conda activation from Codex shells should source:
- `/Users/Mike/lsst/lsstsw/miniconda/etc/profile.d/conda.sh`

Then activate by full path:
```bash
source /Users/Mike/lsst/lsstsw/miniconda/etc/profile.d/conda.sh
conda activate /Users/Mike/miniforge3/envs/py3.12
```

## Sphinx Docs Build
Use this sequence to build docs in Codex:
```bash
source /Users/Mike/lsst/lsstsw/miniconda/etc/profile.d/conda.sh
conda activate /Users/Mike/miniforge3/envs/py3.12
export LC_ALL=C LANG=C
cd /Users/Mike/rmjarvis/TreeCorr/docs
make html
```

Rationale:
- `py3.12` is not always discoverable by name from the conda install on PATH, so full-path
  activation is more reliable.
- Setting `LC_ALL`/`LANG` avoids locale errors when invoking Sphinx.

## TreeCorr Documentation Conventions
- Prefer "data are" (treat "data" as plural) except when clearly used as a mass noun.
- Target a 100-character line length for prose/code comments where practical.
- Do not rewrite or shorten long URLs only to satisfy line-length limits.
- Keep existing documentation images unless there is a concrete issue requiring a change.
- Preserve intentional diction choices (e.g. keep "Contrariwise" where used intentionally).

## Documentation Scope Priorities
- Prioritize user-facing `.rst` pages and Python docstrings in `treecorr/*.py`.
- Config-only docs are lower priority unless explicitly requested.

## Routine Ignore Rules
- Ignore `/Users/Mike/rmjarvis/TreeCorr/docs/_build/` during normal edits/reviews.
- Do not edit `docs/Makefile` just for style/line-length cleanup.

## Typo Sweep Rule
- When a typo is found, grep the whole repo for that same typo pattern (excluding build/data
  artifacts) and fix all clear occurrences in the same pass.

## Linting Preference
- Prefer error-focused lint checks over style linters.
- Preferred Ruff command:
  `ruff check --select F,E9 treecorr`
- This catches likely real issues (undefined names, unused imports/locals, parse errors) without
  style churn.
- Keep intentional exceptions (e.g. package re-exports in `treecorr/__init__.py`) in Ruff config
  (`pyproject.toml` per-file ignores), not inline lint comments.
- Do not keep validation-only code solely to trigger an earlier error if the same error will occur
  naturally when the value/path is actually used.
