# Security

Report a vulnerability through GitHub's private reporting on this
repository, at
<https://github.com/pyscape/utter/security/advisories/new>. It
reaches the maintainer alone and is not visible until a fix is
published.

The runtime reads model files and audio from the host. A model file or
audio that makes it panic, read out of bounds or allocate without bound
is a vulnerability here; a model it refuses to open, naming the part it
does not implement, is not.

## How a report is handled

A report is acknowledged within a few days. The maintainer confirms it,
works a fix privately, and releases it on crates.io and PyPI; the latest
release is the one fixes go to. The reporter is credited in the release
and the advisory unless they ask to stay anonymous, and their details
are kept private throughout.
