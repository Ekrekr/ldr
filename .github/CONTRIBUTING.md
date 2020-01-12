# Contributing

Please [ping me an email](mailto:eliaskassell@gmail.com) if you are interested in contributing.

## Running Tests

- `pytest`. For more info to help debug use `pytest -s`.

## Redeploying

It's not worth doing this for singular small changes or non critical issues.

- Update the version in `setup.py`.

- `python3 setup.py sdist`.

- `twine upload --skip-existing dist/*`.

- Enter credentials.
