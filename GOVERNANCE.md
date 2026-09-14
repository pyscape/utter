# Governance

## Roles

The project has a single maintainer, who reviews and merges changes,
cuts releases, and answers issues and vulnerability reports. Anyone may
contribute through a pull request under [CONTRIBUTING.md](CONTRIBUTING.md).

## How decisions are made

A design decision, meaning architecture, an algorithm, tooling or an
engineering policy, is recorded as a numbered file under
[docs/td](docs/td), described in [docs/td/README.md](docs/td/README.md).
The record states the decision so a reader can tell whether the code
obeys it; it carries no status, date or author. A decision changes by
changing its record, or by a later record that replaces it and says so.
What the library must do for a consumer is stated under `usecases/`.

The maintainer makes the final call on a change. Disagreement is
resolved in the issue or pull request, on the evidence: the decision
records, the gate and benchmark measurements, and the tests.

## Changes to this model

As the project gains maintainers, this file records how roles and
decisions are shared. Until then, the model is as stated above.
