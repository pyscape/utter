# Security

Report a vulnerability through GitHub's private reporting on this
repository: Security, then Report a vulnerability. It reaches the
maintainer alone and is not visible until a fix is published.

The runtime reads model files and audio from the host. A model file or
audio that makes it panic, read out of bounds or allocate without bound
is a vulnerability here; a model it refuses to open, naming the part it
does not implement, is not.

The latest release on crates.io and PyPI is the one fixes go to.
