#![no_main]

use libfuzzer_sys::fuzz_target;
use std::path::PathBuf;
use std::sync::OnceLock;
use utter::{Model, Recognizer};

// The model is a local directory, so the target is a no-op unless UTTER_FUZZ_MODEL names one:
// CI builds it without a model, a local run sets the variable.
fn model() -> Option<&'static Model> {
    static MODEL: OnceLock<Option<Model>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let dir = PathBuf::from(std::env::var_os("UTTER_FUZZ_MODEL")?);
            Model::open(&dir).ok()
        })
        .as_ref()
}

fuzz_target!(|data: &[u8]| {
    let Some(model) = model() else { return };
    let grammar: Vec<String> = ["one", "two", "three", "[unk]"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let Ok(mut rec) = Recognizer::new(model, 16_000.0, &grammar) else {
        return;
    };
    rec.set_words(true);
    rec.set_partial_words(true);
    rec.set_alternatives(3);

    let samples: Vec<i16> = data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    for block in samples.chunks(640) {
        let step = rec.accept(block);
        if step.endpoint {
            let _ = rec.result();
        } else {
            let _ = rec.partial();
        }
    }
    let _ = rec.final_result();
    rec.reset();
    let _ = rec.accept(&samples);
    let _ = rec.final_result();
});
