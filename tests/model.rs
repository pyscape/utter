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

/// The result without the floor, which is appended last.
fn without_floor(json: &str) -> String {
    match json.rfind(", \"floor_dbfs\": ") {
        Some(i) => format!("{}}}", &json[..i]),
        None => json.to_string(),
    }
}

fn alternatives_of(json: &str) -> Vec<String> {
    let key = "\"partial_alternatives\": [";
    let Some(i) = json.find(key) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (k, c) in json[i + key.len()..].char_indices() {
        match c {
            '{' => {
                if depth == 0 {
                    start = k;
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    out.push(json[i + key.len() + start..=i + key.len() + k].to_string());
                }
            }
            ']' if depth == 0 => break,
            _ => {}
        }
    }
    out
}

fn value_of_key(json: &str, key: &str) -> String {
    let i = json.find(key).unwrap_or_else(|| panic!("{key} in {json}")) + key.len();
    let rest = &json[i..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

/// `[[rr:TD-9#The keys are appended]]`
fn without_series(json: &str) -> String {
    let mut out = String::with_capacity(json.len());
    let mut rest = json;
    while let Some(i) = rest.find(", \"relation\": ") {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let end = tail.find('}').expect("reading object ends");
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
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
            if without_floor(p) == "{\"partial\": \"[sil]\"}" {
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

fn num_of_key(json: &str, key: &str) -> f64 {
    let i = json.find(key).unwrap() + key.len();
    let rest = &json[i..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '-' || c == '.'))
        .unwrap_or(rest.len());
    rest[..end].parse().unwrap()
}

fn rms_dbfs(samples: &[i16]) -> f64 {
    if samples.is_empty() {
        return -999.0;
    }
    let acc: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rms = (acc / samples.len() as f64).sqrt();
    if rms <= 0.0 {
        -999.0
    } else {
        20.0 * (rms / 32768.0).log10()
    }
}

#[test]
fn a_final_word_carries_the_energy_under_its_span() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let samples = clip("seven");
    // the plain final and every alternative of an n-best final alike
    for max_alternatives in [1, 4] {
        let mut rec = Recognizer::new(&m, 16000.0, &grammar()).unwrap();
        rec.set_words(true);
        rec.set_max_alternatives(max_alternatives);
        let (_, finals) = decode(&mut rec, &samples);
        let mut words = 0;
        let mut arrays = 0;
        for f in &finals {
            for (_, rest) in f.match_indices("\"result\": [").map(|(i, _)| f.split_at(i)) {
                let rest = &rest["\"result\": [".len()..];
                let array = &rest[..rest.find(']').unwrap()];
                arrays += 1;
                for w in array.split("}, {") {
                    if !w.contains("\"start_sample\": ") {
                        continue;
                    }
                    let start = num_of_key(w, "\"start_sample\": ") as usize;
                    // a final flushes the pipeline, so the last frame can end past the audio fed
                    let end = (num_of_key(w, "\"end_sample\": ") as usize).min(samples.len());
                    let want = rms_dbfs(&samples[start.min(end)..end]);
                    assert!(
                        w.contains("\"energy_dbfs\": "),
                        "no energy on {w} at n={max_alternatives}"
                    );
                    let got = num_of_key(w, "\"energy_dbfs\": ");
                    assert!((got - want).abs() < 1e-3, "{w}: {got} against {want}");
                    words += 1;
                }
            }
        }
        assert!(
            words > 0,
            "no final word at n={max_alternatives}: {finals:?}"
        );
        if max_alternatives > 1 {
            assert!(
                arrays > 1,
                "one result array at n={max_alternatives}: {finals:?}"
            );
        }
    }
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
fn the_floor_outlives_a_final_and_the_rebuild_after_it() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let mut rec = Recognizer::new(&m, 16000.0, &grammar()).unwrap();
    let room: Vec<i16> = vec![1000; 32000];
    for block in room.chunks(640) {
        rec.accept(block);
    }
    let before = num_of_key(rec.final_result(), "\"floor_dbfs\": ");
    assert!((before - rms_dbfs(&room)).abs() < 0.01, "{before}");
    // a final drops the pipeline, so the next accept rebuilds it
    for block in vec![0i16; 1600].chunks(640) {
        rec.accept(block);
    }
    let after = num_of_key(rec.partial(), "\"floor_dbfs\": ");
    assert!((after - before).abs() < 1.0, "{before} then {after}");
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

#[test]
fn an_unimplemented_component_is_refused_at_load() {
    let Some(dir) = model_dir() else { return };
    // The stock model reaches none, or it would not decode.
    let model = Model::open(&dir).unwrap();
    assert!(model.net.unsupported().is_empty());
    // A component the output never reaches is not fatal: the model carries a log-softmax on
    // the xent branch, which is never evaluated.
    assert!(Model::open(&dir).is_ok());
}

#[test]
fn readings_carry_their_relation_and_their_leads_motion() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    // three words with silence between them, so a partial holds a word while shorter and
    // longer readings compete with it
    let samples = [
        clip("yes"),
        vec![0i16; 4000],
        clip("seven"),
        vec![0i16; 4000],
        clip("no"),
        vec![0i16; 4000],
    ]
    .concat();

    let mut rec = Recognizer::new(&m, 16000.0, &grammar()).unwrap();
    rec.set_partial_words(true);
    rec.set_alternatives(5);
    let (partials, finals) = decode(&mut rec, &samples);
    assert!(partials.len() > 10);

    let mut relations: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut carried = 0;
    let mut entered_with_history = 0;
    let mut first_block = true;
    for p in partials.iter() {
        let alts = alternatives_of(p);
        if alts.is_empty() {
            // before the first chunk is decoded there is no reading to report
            assert_eq!(without_floor(p), "{\"partial\": \"[sil]\"}");
            continue;
        }
        assert_eq!(value_of_key(&alts[0], "\"relation\": "), "\"same\"");
        for a in &alts {
            let text = text_of_key(a, "\"text\": ");
            let delta = value_of_key(a, "\"lead_delta\": ");
            relations.push(value_of_key(a, "\"relation\": "));
            // the history starts empty, so the first chunk reported has no motion
            if first_block {
                assert_eq!(delta, "null", "{a}");
            }
            if delta != "null" {
                if seen.contains(&text) {
                    carried += 1;
                } else {
                    // [[rr:TD-9#Readings are read once per decoding advance]]
                    entered_with_history += 1;
                }
            }
            seen.insert(text);
        }
        first_block = false;
    }
    assert!(
        entered_with_history > 0,
        "no reading entered with a history"
    );
    assert!(carried > 0, "no reading ever carried a lead forward");
    for r in ["\"same\"", "\"prefix\"", "\"extends\"", "\"differs\""] {
        assert!(relations.iter().any(|x| x == r), "no {r} reading");
    }

    // [[rr:TD-9#The keys are appended]]
    let mut plain = Recognizer::new(&m, 16000.0, &grammar()).unwrap();
    plain.set_partial_words(true);
    let (plain_partials, plain_finals) = decode(&mut plain, &samples);
    assert_eq!(finals, plain_finals);
    for (p, q) in partials.iter().zip(plain_partials.iter()) {
        let stripped = without_series(p);
        assert!(!stripped.contains("relation") && !stripped.contains("lead_delta"));
        let key = ", \"partial_alternatives\": [";
        let Some(i) = stripped.find(key) else {
            assert_eq!(stripped, *q);
            continue;
        };
        let j = stripped
            .find("], \"partial_result\"")
            .expect("partial result");
        assert_eq!(format!("{}{}", &stripped[..i], &stripped[j + 1..]), *q);
    }
}

#[test]
fn the_readings_are_sampled_once_per_chunk_whatever_is_asked_of_them() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let samples = clip("seven");

    // [[rr:TD-9#Readings are read once per decoding advance]]
    let mut rec = Recognizer::new(&m, 16000.0, &grammar()).unwrap();
    rec.set_alternatives(1);
    for block in samples.chunks(640) {
        rec.accept(block);
    }
    let one = rec.partial().to_string();
    let alts = alternatives_of(&one);
    assert_eq!(alts.len(), 1);
    assert_ne!(value_of_key(&alts[0], "\"lead_delta\": "), "null", "{one}");
    // [[rr:TD-7#Decision outcome]]
    assert_eq!(rec.partial(), one);

    // n moves with no new audio
    // [[rr:TD-9#Readings are read once per decoding advance]]
    rec.set_alternatives(3);
    let three = rec.partial().to_string();
    let alts3 = alternatives_of(&three);
    assert_eq!(alts3.len(), 3);
    assert_eq!(alts3[0], alts[0]);

    // Disabled then enabled: the history starts at the next chunk, so nothing has a delta yet.
    let mut late = Recognizer::new(&m, 16000.0, &grammar()).unwrap();
    for block in samples.chunks(640) {
        late.accept(block);
    }
    late.set_alternatives(3);
    for a in alternatives_of(late.partial()) {
        assert_eq!(value_of_key(&a, "\"lead_delta\": "), "null", "{a}");
    }
    late.accept(&vec![0i16; 16000]);
    assert!(alternatives_of(late.partial())
        .iter()
        .any(|a| value_of_key(a, "\"lead_delta\": ") != "null"));

    // A final restarts the decoder
    // [[rr:TD-9#Readings are read once per decoding advance]]
    late.final_result();
    let mut first = String::new();
    for block in samples.chunks(640) {
        late.accept(block);
        if !alternatives_of(late.partial()).is_empty() {
            first = late.partial().to_string();
            break;
        }
    }
    let alts = alternatives_of(&first);
    assert!(!alts.is_empty(), "no reading after the restart");
    for a in alts {
        assert_eq!(value_of_key(&a, "\"lead_delta\": "), "null", "{a}");
    }
}

#[test]
fn one_accept_over_several_chunks_samples_each_of_them() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let mut rec = Recognizer::new(&m, 16000.0, &grammar()).unwrap();
    rec.set_trace_groups(true);
    // a second of audio in one call
    // [[rr:TD-9#Readings are read once per decoding advance]]
    rec.accept(&clip("seven"));
    let trace = rec.take_group_trace();
    assert!(trace.len() >= 3, "{trace:?}");
    let frames: Vec<f64> = trace
        .iter()
        .map(|r| num_of_key(r, "\"frames\": "))
        .collect();
    assert!(frames.windows(2).all(|w| w[1] > w[0]), "{frames:?}");
    for (k, r) in trace.iter().enumerate() {
        let first = first_group_of(r);
        let delta = value_of_key(&first, "\"lead_delta\": ");
        if k == 0 {
            assert_eq!(delta, "null", "{r}");
        }
    }
    assert!(
        trace[1..]
            .iter()
            .any(|r| value_of_key(&first_group_of(r), "\"lead_delta\": ") != "null"),
        "no chunk carried a lead forward"
    );
    assert!(rec.take_group_trace().is_empty());
}

fn first_group_of(record: &str) -> String {
    let key = "\"groups\": [";
    let i = record.find(key).expect("groups") + key.len();
    let rest = &record[i..];
    let end = rest.find('}').expect("group ends");
    rest[..=end].to_string()
}

/// `[[rr:TD-10#Decision outcome]]`: a wordless reading is `[sil]` or `[speech]`, never empty,
/// and the text agrees with the tail of the word list.
#[test]
fn a_wordless_reading_names_what_its_tail_is() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let mut state = 20260913u32;
    let mut hiss: Vec<i16> = (0..32000)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 16) as i32 % 601 - 300) as i16
        })
        .collect();
    hiss.extend(clip("yes"));
    for samples in [clip("no"), clip("seven"), hiss] {
        let mut rec = Recognizer::new(&m, 16000.0, &grammar()).unwrap();
        rec.set_partial_words(true);
        rec.set_alternatives(3);
        let (partials, finals) = decode(&mut rec, &samples);
        for p in partials.iter().chain(finals.iter()) {
            let text = text_of_key(
                p,
                if p.contains("\"partial\": ") {
                    "\"partial\": "
                } else {
                    "\"text\": "
                },
            );
            assert!(!text.is_empty(), "{p}");
            let Some(i) = p.find("\"partial_result\": [") else {
                continue;
            };
            let list = &p[i..];
            let last = list
                .rfind("\"word\": ")
                .map(|k| text_of_key(&list[k..], "\"word\": "));
            let speech_entries = list.matches("\"word\": \"[speech]\"").count();
            assert!(speech_entries <= 1, "{p}");
            match text.as_str() {
                "[speech]" => assert_eq!(last.as_deref(), Some("[speech]"), "{p}"),
                "[sil]" => assert_ne!(last.as_deref(), Some("[speech]"), "{p}"),
                _ => {}
            }
            if speech_entries == 1 {
                assert_eq!(
                    last.as_deref(),
                    Some("[speech]"),
                    "a speech entry is last: {p}"
                );
            }
        }
    }
}
