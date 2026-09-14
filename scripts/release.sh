#!/usr/bin/env bash
# Cut a release: check every precondition, sign and verify the tag, push it, watch the workflow,
# and confirm what it produced. Stops at the first thing that is not as it should be.
#
#     scripts/release.sh            # the version in Cargo.toml
#     scripts/release.sh --check    # the preconditions only, no tag
set -euo pipefail

check_only=false
[ "${1:-}" = "--check" ] && check_only=true

fail() { printf 'release: %s\n' "$*" >&2; exit 1; }
ok() { printf '  ok  %s\n' "$*"; }

cd "$(git rev-parse --show-toplevel)"

version=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -1)
[ -n "$version" ] || fail "no version in Cargo.toml"
tag="v$version"
printf 'release %s\n' "$tag"

[ "$(git branch --show-current)" = main ] || fail "not on main"
ok "on main"
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "tracked files have uncommitted changes"
ok "tree clean"
git fetch -q origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || fail "main is not at origin/main; pull or push first"
ok "main matches origin"
[ -f "docs/release-notes/$tag.md" ] || fail "docs/release-notes/$tag.md is missing; the workflow reads the release body from it"
ok "release notes exist"
grep -q "^## $version\$" CHANGELOG.md || fail "CHANGELOG.md has no '## $version' heading"
ok "changelog heading"
grep -q '^## Unreleased' CHANGELOG.md && fail "CHANGELOG.md still has an Unreleased section"
git rev-parse -q --verify "refs/tags/$tag" >/dev/null && fail "tag $tag already exists locally"
git ls-remote --exit-code --tags origin "$tag" >/dev/null 2>&1 && fail "tag $tag already exists on origin"
ok "tag $tag is free"
[ "$(git config --get tag.gpgSign)" = true ] || fail "tag.gpgSign is not on; the tag must be signed"
ok "tags are signed"
code=$(curl -s -o /dev/null -w '%{http_code}' -A "utter release script" "https://crates.io/api/v1/crates/utter/$version")
[ "$code" = 404 ] || fail "crates.io answers $code for utter $version; expected 404 before publishing"
ok "crates.io has no $version"

if $check_only; then
    printf 'preconditions hold; run without --check to tag\n'
    exit 0
fi

git tag -s "$tag" -m "utter $version"
git tag -v "$tag"
ok "tag signed and verified"
git push origin "$tag"
ok "tag pushed"

sleep 10
run=$(gh run list --workflow release.yml --branch "$tag" --limit 1 --json databaseId -q '.[0].databaseId')
[ -n "$run" ] || fail "no release.yml run found for $tag"
gh run watch "$run" --exit-status || fail "release workflow failed; see gh run view $run --log-failed"
ok "workflow succeeded"

draft=$(gh release view "$tag" --json isDraft -q '.isDraft')
[ "$draft" = false ] || fail "release $tag is a draft"
assets=$(gh release view "$tag" --json assets -q '.assets[].name')
n=$(printf '%s\n' "$assets" | grep -c .)
printf '%s\n' "$assets" | sed 's/^/      /'
[ "$n" -eq 6 ] || fail "expected 6 assets (five archives and the Sigstore bundle), found $n"
printf '%s\n' "$assets" | grep -q "^utter-$tag.sigstore.json\$" || fail "the Sigstore bundle is not among the assets"
ok "release has six assets and is published"

for _ in 1 2 3 4 5 6; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -A "utter release script" "https://crates.io/api/v1/crates/utter/$version")
    [ "$code" = 200 ] && break
    sleep 10
done
[ "$code" = 200 ] || fail "crates.io answers $code for utter $version after the workflow"
ok "crates.io has utter $version"
printf 'released %s\n' "$tag"
