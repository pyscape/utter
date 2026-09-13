//! The revision the crate was built from, so a binding can say which runtime it carries.
//! Shelling out to git rather than taking a crate for it: `[[rr:TD-2#Dependency policy]]`.
//! Degrades to "unknown" wherever git or the checkout is absent (a packaged crate, a
//! vendored copy).

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
        None => "unknown".to_string(),
    };
    println!("cargo:rustc-env=UTTER_REVISION={revision}");
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
