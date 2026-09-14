# Contributing

## Proposing a change

Open an issue for anything larger than a small fix, so the approach can
be agreed before the work. Changes reach `main` through a pull request;
`main` is protected, and a request merges only when every check has
passed and the branch is current.

A design decision, meaning architecture, an algorithm, tooling or an
engineering policy, is recorded under [docs/td](docs/td), one numbered
file per decision. [docs/td/README.md](docs/td/README.md) says how; a
change that makes such a decision carries its record.

## What a change must satisfy

The checks below run in CI and each one blocks a merge. Run them before
opening the request:

- `cargo fmt --check`
- `cargo clippy --release --all-targets -- -D warnings`
- `cargo test` and `cargo test --release`, and, against a stock model,
  `UTTER_TEST_MODEL=<model dir> cargo test --release`
- `cargo doc --no-deps` with `RUSTDOCFLAGS=-D warnings`
- `rr verify` for the decision-record markers (see the rr reference
  under `docs/`)
- `ruff check scripts`, `ruff format --check scripts` and `mypy` for
  the Python
- `typos`

New functionality comes with tests. CI measures line coverage against a
floor, so a change that adds untested code fails the build.

Comment only what the code cannot say. A comment that restates the code
is removed; a rule that lives elsewhere is cited, not repeated.

## Certifying origin

Contributions are certified under the Developer Certificate of Origin,
<https://developercertificate.org>: by signing off a commit you assert
you wrote the change or may submit it under the project's license. Add
the line with `git commit -s`:

```text
Signed-off-by: Your Name <you@example.com>
```

## License

The crate is MIT, with the vendored Vosk-Rust files kept under
Apache-2.0. A contribution is offered under those same terms.

## Releasing

`scripts/release.sh --check` says whether a release can be cut from
the checkout: on main and level with origin, a clean tree, the version
in Cargo.toml with its `docs/release-notes/vX.Y.Z.md` and CHANGELOG
heading, no Unreleased section, no such tag yet. Without `--check` it
signs and verifies the tag, pushes it, watches the release workflow,
and confirms the seven assets and the crate on crates.io.
