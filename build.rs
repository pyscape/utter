//! The revision the crate was built from, so a binding can say which runtime it carries.
//! Shelling out to git rather than taking a crate for it: `[[rr:TD-2#Dependency policy]]`.
//! A packaged crate has no checkout but carries the commit it was packaged from in
//! `.cargo_vcs_info.json`, which cargo writes into every package made from a git tree; the
//! revision degrades to "unknown" only where neither exists (a vendored copy, a tarball).

use std::process::Command;

fn main() {
    let head = run(&["rev-parse", "HEAD"]);
    let revision = match head {
        Some(h) => {
            let dirty = run(&["status", "--porcelain", "--untracked-files=no"])
                .map(|s| !s.is_empty())
                .unwrap_or(false);
            if dirty {
                format!("{h}-dirty")
            } else {
                h
            }
        }
        None => packaged_revision().unwrap_or_else(|| "unknown".to_string()),
    };
    println!("cargo:rustc-env=UTTER_REVISION={revision}");
    println!("cargo:rerun-if-changed=.cargo_vcs_info.json");
    // A commit or a working-tree edit changes the constant, and nothing else does.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
    if let Some(r) = run(&["symbolic-ref", "--quiet", "HEAD"]) {
        println!("cargo:rerun-if-changed=.git/{r}");
    }
}

fn run(args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The commit recorded by `cargo package`: `{"git":{"sha1":"<hex>"[,"dirty":true]},...}`.
/// Read by hand for the same reason git is shelled out to.
fn packaged_revision() -> Option<String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/.cargo_vcs_info.json");
    let text = std::fs::read_to_string(path).ok()?;
    let sha1 = text.split("\"sha1\"").nth(1)?.split('"').nth(1)?;
    if sha1.len() != 40 || !sha1.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let dirty = text
        .split("\"dirty\"")
        .nth(1)
        .is_some_and(|rest| rest.trim_start_matches([':', ' ']).starts_with("true"));
    Some(if dirty {
        format!("{sha1}-dirty")
    } else {
        sha1.to_string()
    })
}
