//! Tests against the stock small English Vosk model: skipped unless `UTTER_TEST_MODEL` names its
//! directory. CI downloads the model; locally, point the variable at a copy.

use std::path::PathBuf;
use utter::wav::read_wav;
use utter::{Model, Recognizer};

fn model_dir() -> Option<PathBuf> {
    std::env::var_os("UTTER_TEST_MODEL")
        .map(PathBuf::from)
        .filter(|p| p.join("am/final.mdl").exists())
}

fn grammar() -> Vec<String> {
    [
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "one", "two",
        "three", "four", "five", "six", "seven", "eight", "nine", "zero",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn clip(name: &str) -> Vec<i16> {
    read_wav(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(format!("{name}.wav")),
    )
    .unwrap()
    .samples
}

fn decode(rec: &mut Recognizer, samples: &[i16]) -> (Vec<String>, Vec<String>) {
    let mut partials = Vec::new();
    let mut finals = Vec::new();
    for block in samples.chunks(640) {
        if rec.accept(block).endpoint {
            finals.push(rec.result().to_string());
        } else {
            partials.push(rec.partial().to_string());
        }
    }
    finals.push(rec.final_result().to_string());
    (partials, finals)
}

fn text_of(json: &str) -> String {
    let key = "\"text\": ";
    let i = json.rfind(key).expect("text key") + key.len();
    let rest = &json[i + 1..];
    rest[..rest.find('"').unwrap()].to_string()
}

#[test]
fn model_opens_and_reports_its_shape() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    assert_eq!(m.net.input_dim, 40);
    assert!(m.net.ivector_dim > 0);
    assert!(m.tm.num_pdfs > 0 && m.net.output_dim == m.tm.num_pdfs);
    assert!(m.hcl.num_states() > 1000);
    assert_eq!(m.reach.len(), m.hcl.num_states());
    assert!(m.word_ids.contains_key("yes") && m.unknown_word().is_some());
    assert!(m.ivector.is_some());
}

#[test]
fn clips_decode_to_their_words() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let mut rec = Recognizer::new(&m, 16000.0, &grammar()).unwrap();
    rec.set_words(true);
    rec.set_partial_words(true);
    rec.set_alternatives(3);
    for word in ["yes", "no", "seven"] {
        let (partials, finals) = decode(&mut rec, &clip(word));
        let texts: Vec<String> = finals.iter().map(|f| text_of(f)).collect();
        assert!(texts.iter().any(|t| t == word), "{word}: finals {texts:?}");
        // every partial has rank 0 equal to its text and carries the extended keys
        for p in &partials {
            assert!(p.starts_with("{\"partial\": "), "{p}");
            if p == "{\"partial\": \"[sil]\"}" {
                // nothing decoded yet on this block
                continue;
            }
            assert!(
                p.contains("\"partial_alternatives\": [") && p.contains("\"partial_result\": [")
            );
            let partial = text_of_key(p, "\"partial\": ");
            let rank0 = text_of_key(
                &p[p.find("\"partial_alternatives\"").unwrap()..],
                "\"text\": ",
            );
            assert_eq!(partial, rank0);
        }
        assert!(
            partials
                .iter()
                .any(|p| p.contains(&format!("\"partial\": \"{word}\""))),
            "{word} never in a partial"
        );
        // finals carry conf 1.0, sample intervals, and the word
        let with_word = finals.iter().find(|f| text_of(f) == word).unwrap();
        assert!(with_word.contains("\"conf\": 1.0") && with_word.contains("\"start_sample\": "));
    }
}

fn text_of_key(json: &str, key: &str) -> String {
    let i = json.find(key).unwrap() + key.len();
    let rest = &json[i + 1..];
    rest[..rest.find('"').unwrap()].to_string()
}

#[test]
fn silence_reads_as_sil_and_the_stream_is_deterministic() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let quiet = vec![0i16; 16000];
    let mut rec = Recognizer::new(&m, 16000.0, &grammar()).unwrap();
    rec.set_alternatives(2);
    let (partials, finals) = decode(&mut rec, &quiet);
    assert!(
        partials
            .iter()
            .all(|p| p.starts_with("{\"partial\": \"[sil]\"")),
        "{:?}",
        partials.last()
    );
    assert_eq!(text_of(finals.last().unwrap()), "[sil]");
    let samples = clip("yes");
    let mut a = Recognizer::new(&m, 16000.0, &grammar()).unwrap();
    let mut b = Recognizer::new(&m, 16000.0, &grammar()).unwrap();
    assert_eq!(decode(&mut a, &samples), decode(&mut b, &samples));
}

#[test]
fn grammar_bounds_and_unknown_option() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    assert!(Recognizer::new(&m, 16000.0, &[]).is_err());
    let big: Vec<String> = (0..300).map(|i| format!("w{i}")).collect();
    assert!(Recognizer::new(&m, 16000.0, &big).is_err());
    let opts = utter::recognizer::RecognizerOptions {
        unknown_cost: Some(0.0),
        ..Default::default()
    };
    let rec = Recognizer::with_options(&m, 16000.0, &grammar(), &opts).unwrap();
    assert!(rec.graph().num_states() > 0);
}

#[test]
fn c_abi_round_trip() {
    let Some(dir) = model_dir() else { return };
    use std::ffi::{CStr, CString};
    use utter::capi::*;
    let path = CString::new(dir.to_str().unwrap()).unwrap();
    let grammar = CString::new("[\"yes\", \"no\", \"seven\"]").unwrap();
    unsafe {
        let model = utter_model_new(path.as_ptr());
        assert!(!model.is_null());
        let yes = CString::new("yes").unwrap();
        assert!(utter_model_find_word(model, yes.as_ptr()) > 0);
        let rec = utter_recognizer_new_grm(model, 16000.0, grammar.as_ptr());
        assert!(!rec.is_null());
        utter_recognizer_set_words(rec, 1);
        let samples = clip("no");
        let mut endpoints = 0;
        for block in samples.chunks(640) {
            let r = utter_recognizer_accept_waveform_s(rec, block.as_ptr(), block.len() as i32);
            assert!(r >= 0);
            endpoints += r;
        }
        let fin = CStr::from_ptr(utter_recognizer_final_result(rec))
            .to_str()
            .unwrap()
            .to_string();
        assert!(fin.contains("\"text\": \"no\"") || endpoints > 0, "{fin}");
        assert!(utter_recognizer_decoded_sample(rec) > 0);
        utter_recognizer_free(rec);
        utter_model_free(model);
    }
}

#[test]
fn one_grammar_is_composed_once_and_the_cache_has_a_bound() {
    let Some(dir) = model_dir() else { return };
    let model = Model::open(&dir).unwrap();
    let g = grammar();
    let first = Recognizer::new(&model, 16000.0, &g).unwrap();
    let second = Recognizer::new(&model, 16000.0, &g).unwrap();
    assert!(
        std::ptr::eq(first.graph(), second.graph()),
        "a second recognizer on the same grammar composed its own graph"
    );
    // A different grammar is a different graph, and enough of them evict the first.
    for word in g.iter().take(utter::model::CACHED_GRAPHS) {
        let r = Recognizer::new(&model, 16000.0, std::slice::from_ref(word)).unwrap();
        assert!(!std::ptr::eq(first.graph(), r.graph()));
    }
    let again = Recognizer::new(&model, 16000.0, &g).unwrap();
    assert!(
        !std::ptr::eq(first.graph(), again.graph()),
        "the cache kept more grammars than CACHED_GRAPHS"
    );
    assert_eq!(first.graph().num_states(), again.graph().num_states());
}

#[test]
fn silence_weighting_can_be_turned_off_for_the_kaldi_comparison() {
    let Some(dir) = model_dir() else { return };
    let model = Model::open(&dir).unwrap();
    let opts = utter::recognizer::RecognizerOptions {
        silence_weight: 1.0,
        ..Default::default()
    };
    let mut rec = Recognizer::with_options(&model, 16000.0, &grammar(), &opts).unwrap();
    let (_, finals) = decode(&mut rec, &clip("yes"));
    assert_eq!(text_of(finals.last().unwrap()), "yes");
}
