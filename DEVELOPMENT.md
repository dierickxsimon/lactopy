# Development

## Project structure
The basic project structure looks like this:
```
.
├── docs -------------------------------- (all documentation with mkdocs)
├── notebooks --------------------------- (some notebooks to fidget around with some of the code)
├── src
│   └── lactopy
|       └── plots
|           └── __init__.py -------------- (All plot related code)
|       └── lactatemodels
|           └── __init__.py -------------- (All model related code)
│       ├── __init__.py
│       └── py.typed
├── tests
│   ├── data
│   │   └── {test files needed for tests}
│   └── {test_*.py files}
└── uv.lock
```

The main thing to note is that all the code is in the `src` directory, and all the tests are in the `tests` directory.


## Dependencies

Dependencies are managed in the `pyproject.toml` file and managed with `uv`([uv](https://docs.astral.sh/uv/)).

To add dependencies, run `uv add <dependency>`.
To add a development dependency, run `uv add --dev <dependency>`.


## Testing

To run tests, run either `uv run pytest`.



## Git workflow

- Never push to `main` directly
- Create feature branches named `feature/{initials}_{feature_slug}` (e.g. `git checkout -b feature/sd_my_awesome_feature`)
- Create a Pull Request to `main` when ready
- Try not to merge your own PRs (we might enforce stricter rules later but for now let's keep it simple and fast)


## Pre-PR checklist

- [ ] Squash commits (be reasonable)
- [ ] Update `CHANGELOG.md`
- [ ] All tests pass


## Pre-commit hooks

To install the pre-commit hooks run the command `uv run pre-commit install`. Now `pre-commit` will run automatically on `git commit`.


## Documentation
To run the documentation locally run following command:

```bash
uv run mkdocs serve
```
