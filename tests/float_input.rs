//! The float entry point against the integer one, on the stock small English model: skipped
//! unless `UTTER_TEST_MODEL` names its directory, with the speaker evidence compared as well when
//! `UTTER_TEST_SPK_MODEL` names the speaker model's.
// [[rr:TD-13#Compatibility is verified at the input and result boundaries]]

// The test signals are generated from sample counts and amplitudes.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::path::PathBuf;
use utter::wav::read_wav;
use utter::{AudioInputErrorKind, Model, Recognizer, SpeakerModel, Step};

fn model_dir() -> Option<PathBuf> {
    std::env::var_os("UTTER_TEST_MODEL")
        .map(PathBuf::from)
        .filter(|p| p.join("am/final.mdl").exists())
}

fn spk_dir() -> Option<PathBuf> {
    std::env::var_os("UTTER_TEST_SPK_MODEL")
        .map(PathBuf::from)
        .filter(|p| p.join("final.ext.raw").exists())
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

fn wide() -> Vec<String> {
    let mut wide = grammar();
    wide.extend("abcdefghijklmnopqrstuvwxyz".chars().map(|c| c.to_string()));
    wide.extend(
        [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
            "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
            "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    wide
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

/// `[[rr:TD-13#The connection carries normalized waveform samples]]`
fn normalized(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| f32::from(s) / 32768.0).collect()
}

/// Uniform noise at a level, from a fixed seed, as a quiet room's hiss.
fn room(dbfs: f64, seconds: f64) -> Vec<i16> {
    let peak = 32768.0 * 10f64.powf(dbfs / 20.0) * 3f64.sqrt();
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..(16000.0 * seconds) as usize)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as f64 / f64::from(1u32 << 31);
            ((u * 2.0 - 1.0) * peak).round() as i16
        })
        .collect()
}

/// Words, pauses of digital silence and of hiss, long enough between them for the model's rules,
/// the host's bound and its floor margin each to close a final.
fn speech_and_pauses() -> Vec<i16> {
    [
        clip("yes"),
        room(-50.0, 1.5),
        clip("seven"),
        vec![0; 12000],
        clip("no"),
        room(-55.0, 3.0),
        clip("seven"),
        room(-50.0, 6.0),
    ]
    .concat()
}

/// Under the wide grammar of the public pages; under a small one the hiss reads as a word.
fn recognizer<'m>(
    model: &'m Model,
    spk: Option<&'m SpeakerModel>,
    veto: Option<f32>,
) -> Recognizer<'m> {
    let mut rec = Recognizer::new(model, 16000.0, &wide()).unwrap();
    rec.set_words(true);
    rec.set_partial_words(true);
    rec.set_alternatives(3);
    rec.set_endpoint_bound(Some(300.0), veto);
    rec.set_endpoint_floor_margin(Some(8.0));
    rec.set_spk_model(spk).unwrap();
    rec
}

fn num_of_key(json: &str, key: &str) -> Option<f64> {
    let i = json.find(key)? + key.len();
    let rest = &json[i..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '-' || c == '.'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn endpoint_of(json: &str) -> String {
    let key = "\"endpoint\": \"";
    json.find(key).map_or(String::new(), |i| {
        let rest = &json[i + key.len()..];
        rest[..rest.find('"').unwrap()].to_string()
    })
}

#[derive(Clone, Copy)]
enum Via {
    Integer,
    Float,
}

fn feed(rec: &mut Recognizer, block: &[i16], via: Via) -> Step {
    match via {
        Via::Integer => rec.accept(block),
        Via::Float => rec.accept_f32(&normalized(block)).unwrap(),
    }
}

fn same_step(a: Step, b: Step, at: &str) {
    assert_eq!((a.endpoint, a.sample), (b.endpoint, b.sample), "{at}");
}

/// Feed `audio` in blocks to `a` as integers and to `b` by `via`, run `between` after every
/// block, and require after each the same step, features and partial or final, and at the end
/// the same flush. Returns every final, the flush last.
fn lockstep(
    a: &mut Recognizer,
    b: &mut Recognizer,
    audio: &[i16],
    block: usize,
    via: impl Fn(usize) -> Via,
    mut between: impl FnMut(usize, Step, &mut Recognizer, &mut Recognizer),
) -> Vec<String> {
    let mut finals = Vec::new();
    for (i, x) in audio.chunks(block).enumerate() {
        let at = format!("block {i} of {block}");
        let (p, q) = (a.accept(x), feed(b, x, via(i)));
        same_step(p, q, &at);
        assert!(a.features() == b.features(), "features after {at}");
        if p.endpoint {
            let r = a.result().to_string();
            assert_eq!(r, b.result(), "final after {at}");
            finals.push(r);
        } else {
            assert_eq!(a.partial(), b.partial(), "partial after {at}");
        }
        between(i, p, a, b);
    }
    let r = a.final_result().to_string();
    assert_eq!(r, b.final_result(), "flush at {block}");
    finals.push(r);
    finals
}

#[test]
fn integer_samples_as_floats_give_what_integers_give() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let spk = spk_dir().map(|d| SpeakerModel::open(&d).unwrap());
    let audio = speech_and_pauses();
    let mut endpoints = std::collections::BTreeSet::new();
    for (block, veto) in [
        (160, None),
        (640, Some(8.0)),
        (1000, None),
        (audio.len(), None),
    ] {
        let (mut a, mut b) = (
            recognizer(&m, spk.as_ref(), veto),
            recognizer(&m, spk.as_ref(), veto),
        );
        let finals = lockstep(
            &mut a,
            &mut b,
            &audio,
            block,
            |_| Via::Float,
            |_, _, _, _| {},
        );
        endpoints.extend(finals.iter().map(|f| endpoint_of(f)));
        if spk.is_some() {
            assert!(
                finals.iter().any(|f| f.contains("\"spk\": [")),
                "{finals:?}"
            );
        }
    }
    for kind in ["rule1", "rule2", "bound", "floor", "flush"] {
        assert!(endpoints.contains(kind), "no {kind} final: {endpoints:?}");
    }
}

#[test]
fn the_two_entry_points_interleave_on_one_recognizer() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let spk = spk_dir().map(|d| SpeakerModel::open(&d).unwrap());
    let audio = speech_and_pauses();
    let (mut a, mut b) = (
        recognizer(&m, spk.as_ref(), None),
        recognizer(&m, spk.as_ref(), None),
    );
    let via = |i: usize| {
        if i.is_multiple_of(3) {
            Via::Integer
        } else {
            Via::Float
        }
    };
    let finals = lockstep(&mut a, &mut b, &audio, 640, via, |_, _, _, _| {});
    assert!(finals.len() > 2, "{finals:?}");
}

#[test]
fn an_empty_block_keeps_the_lifecycle_and_does_not_move_the_clock() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let (mut a, mut b) = (recognizer(&m, None, None), recognizer(&m, None, None));
    same_step(a.accept(&[]), b.accept_f32(&[]).unwrap(), "first call");
    assert_eq!(a.partial(), b.partial());
    let audio = speech_and_pauses();
    let mut after_final = 0;
    let empty = |i: usize, p: Step, a: &mut Recognizer, b: &mut Recognizer| {
        if p.endpoint || i % 5 == 4 {
            let (q, r) = (a.accept(&[]), b.accept_f32(&[]).unwrap());
            same_step(q, r, &format!("empty call after block {i}"));
            assert_eq!(
                r.sample, p.sample,
                "an empty call after block {i} moved the clock"
            );
            assert_eq!(a.partial(), b.partial());
            after_final += usize::from(p.endpoint);
        }
    };
    lockstep(&mut a, &mut b, &audio, 640, |_| Via::Float, empty);
    assert!(after_final > 0);
    // after the flush an empty call rebuilds the pipeline as the integer one does
    same_step(a.accept(&[]), b.accept_f32(&[]).unwrap(), "after the flush");
    assert_eq!(a.partial(), b.partial());
    lockstep(
        &mut a,
        &mut b,
        &clip("yes"),
        640,
        |_| Via::Float,
        |_, _, _, _| {},
    );
}

#[test]
fn a_refused_block_changes_nothing() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let spk = spk_dir().map(|d| SpeakerModel::open(&d).unwrap());
    let audio = speech_and_pauses();
    let (mut a, mut b) = (
        recognizer(&m, spk.as_ref(), Some(8.0)),
        recognizer(&m, spk.as_ref(), Some(8.0)),
    );
    let bad = [
        (f32::NAN, AudioInputErrorKind::NotANumber),
        (f32::INFINITY, AudioInputErrorKind::Infinite),
        (f32::NEG_INFINITY, AudioInputErrorKind::Infinite),
        (1.0 + f32::EPSILON, AudioInputErrorKind::OutOfRange),
        (-1.5, AudioInputErrorKind::OutOfRange),
    ];
    let refuse = |b: &mut Recognizer, n: usize, from: &[i16]| {
        let (value, kind) = bad[n % bad.len()];
        // five seconds of good audio and one bad sample at its end
        let mut block = normalized(&from[..from.len().min(80000)]);
        block.push(value);
        let e = b.accept_f32(&block).unwrap_err();
        assert_eq!((e.index, e.kind), (block.len() - 1, kind));
    };
    refuse(&mut b, 0, &audio);
    let mut n = 0;
    let finals = lockstep(
        &mut a,
        &mut b,
        &audio,
        640,
        |_| Via::Float,
        |i, p, _, b| {
            if p.endpoint || i % 4 == 1 {
                n += 1;
                refuse(b, n, &audio[(i * 640).min(audio.len())..]);
            }
        },
    );
    assert!(finals.len() > 2, "{finals:?}");
    refuse(&mut b, 3, &audio);
    lockstep(
        &mut a,
        &mut b,
        &clip("no"),
        640,
        |_| Via::Float,
        |_, _, _, _| {},
    );
}

/// The front-end features kept and a partial with its word list, of a recognizer fed a second
/// of a signal in 40 ms blocks; `feed` is handed each block's sample indices.
fn level_run(m: &Model, feed: impl Fn(&mut Recognizer, &[f32])) -> (Vec<Vec<f32>>, String) {
    let mut rec = Recognizer::new(m, 16000.0, &grammar()).unwrap();
    rec.set_partial_words(true);
    let index: Vec<f32> = (0..16000).map(|i| i as f32).collect();
    for block in index.chunks(640) {
        feed(&mut rec, block);
    }
    (rec.features().0.to_vec(), rec.partial().to_string())
}

fn tone(amplitude: f32, i: f32) -> f32 {
    amplitude * (std::f32::consts::TAU * 440.0 * i / 16000.0).sin()
}

#[test]
fn a_fraction_of_a_count_reaches_the_features_and_the_energy() {
    let Some(dir) = model_dir() else { return };
    let mut m = Model::open(&dir).unwrap();
    m.mfcc_opts.dither = 0.0;
    // 0.4 of a 16-bit count at the peak: every sample rounds to zero as an integer
    let quiet = 0.4 / 32768.0;
    let (quiet_features, quiet_partial) = level_run(&m, |rec, t| {
        let x: Vec<f32> = t.iter().map(|&i| tone(quiet, i)).collect();
        rec.accept_f32(&x).unwrap();
    });
    let (zero_features, zero_partial) = level_run(&m, |rec, t| {
        let x: Vec<i16> = t
            .iter()
            .map(|&i| (tone(quiet, i) * 32768.0).round() as i16)
            .collect();
        assert!(x.iter().all(|&s| s == 0));
        rec.accept(&x);
    });
    assert!(quiet_features != zero_features);
    assert_eq!(quiet_features.len(), zero_features.len());
    let want = 20.0 * (f64::from(quiet) / 2f64.sqrt()).log10();
    let floor = num_of_key(&quiet_partial, "\"floor_dbfs\": ").unwrap();
    assert!((floor - want).abs() < 0.05, "{floor} against {want}");
    let energy = num_of_key(&quiet_partial, "\"energy_dbfs\": ").unwrap();
    assert!((energy - want).abs() < 0.05, "{quiet_partial}");
    assert!(!zero_partial.contains("floor_dbfs"), "{zero_partial}");
    assert!(
        zero_partial.contains("\"energy_dbfs\": null"),
        "{zero_partial}"
    );
}

#[test]
fn full_scale_is_one_and_reads_zero_dbfs() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let floor_of = |f: &dyn Fn(f32) -> f32| {
        let (_, partial) = level_run(&m, |rec, t| {
            let x: Vec<f32> = t.iter().map(|&i| f(i)).collect();
            rec.accept_f32(&x).unwrap();
        });
        num_of_key(&partial, "\"floor_dbfs\": ").unwrap()
    };
    assert_eq!(floor_of(&|_| 1.0), 0.0);
    assert_eq!(floor_of(&|_| -1.0), 0.0);
    assert!((floor_of(&|_| 0.5) - 20.0 * 0.5f64.log10()).abs() < 1e-6);
    let want = 20.0 * (0.25 / 2f64.sqrt()).log10();
    assert!((floor_of(&|i| tone(0.25, i)) - want).abs() < 0.01);
}

#[test]
fn digital_silence_has_no_level_through_either_entry_point() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    let (integer_features, integer) = level_run(&m, |rec, t| {
        rec.accept(&vec![0; t.len()]);
    });
    for zero in [0.0, -0.0] {
        let (features, float) = level_run(&m, |rec, t| {
            rec.accept_f32(&vec![zero; t.len()]).unwrap();
        });
        assert!(features == integer_features);
        assert_eq!(float, integer);
    }
    assert!(!integer.contains("floor_dbfs"), "{integer}");
    assert!(integer.contains("\"energy_dbfs\": null"), "{integer}");
}

const HOP: usize = 160;
const BLOCK: usize = 640;

/// A stand-in for a waveform processor such as a speaker extractor: it hands back the waveform it
/// is given, normalized, in 160-sample hops once 480 samples of lookahead stand behind a hop, and
/// on `finish` hands back all it still holds, the last hop short.
#[derive(Default)]
struct Upstream {
    held: Vec<f32>,
}

impl Upstream {
    const LOOKAHEAD: usize = 480;

    fn push(&mut self, capture: &[i16], mut out: impl FnMut(&[f32])) {
        self.held.extend(normalized(capture));
        while self.held.len() >= Self::LOOKAHEAD + HOP {
            out(&self.held[..HOP]);
            self.held.drain(..HOP);
        }
    }

    fn finish(&mut self, mut out: impl FnMut(&[f32])) {
        for hop in self.held.chunks(HOP) {
            out(hop);
        }
        self.held.clear();
    }
}

/// The host between the two: each hop straight to the recognizer, or gathered into a fixed 40 ms
/// block that goes once full. Keeps every call's length and what the recognizer said after it.
struct Host<'m> {
    rec: Recognizer<'m>,
    gather: bool,
    block: [f32; BLOCK],
    filled: usize,
    calls: Vec<usize>,
    said: Vec<String>,
}

impl<'m> Host<'m> {
    fn new(model: &'m Model, gather: bool) -> Self {
        let mut rec = Recognizer::new(model, 16000.0, &grammar()).unwrap();
        rec.set_words(true);
        Host {
            rec,
            gather,
            block: [0.0; BLOCK],
            filled: 0,
            calls: Vec::new(),
            said: Vec::new(),
        }
    }

    fn send(&mut self, samples: &[f32]) {
        let step = self.rec.accept_f32(samples).unwrap();
        self.calls.push(samples.len());
        let said = if step.endpoint {
            self.rec.result()
        } else {
            self.rec.partial()
        };
        self.said.push(said.to_string());
    }

    fn hop(&mut self, mut hop: &[f32]) {
        if !self.gather {
            return self.send(hop);
        }
        while !hop.is_empty() {
            let n = hop.len().min(BLOCK - self.filled);
            self.block[self.filled..self.filled + n].copy_from_slice(&hop[..n]);
            self.filled += n;
            hop = &hop[n..];
            if self.filled == BLOCK {
                let block = self.block;
                self.send(&block);
                self.filled = 0;
            }
        }
    }

    /// A segment of capture, 50 ms at a time, through a fresh processor; at its end the processor
    /// is drained, then the part-filled block sent, then the recognizer flushed.
    fn segment(&mut self, capture: &[i16]) {
        let mut up = Upstream::default();
        for c in capture.chunks(800) {
            up.push(c, |hop| self.hop(hop));
        }
        up.finish(|hop| self.hop(hop));
        if self.filled > 0 {
            let block = self.block;
            self.send(&block[..self.filled]);
            self.filled = 0;
        }
        self.said.push(self.rec.final_result().to_string());
    }
}

/// The same blocks as 16-bit samples through `accept`, and what the recognizer said.
fn replay(model: &Model, capture: &[i16], calls: &[usize]) -> Vec<String> {
    let mut rec = Recognizer::new(model, 16000.0, &grammar()).unwrap();
    rec.set_words(true);
    let mut said = Vec::new();
    let mut at = 0;
    for &n in calls {
        let step = rec.accept(&capture[at..at + n]);
        at += n;
        said.push(
            if step.endpoint {
                rec.result()
            } else {
                rec.partial()
            }
            .to_string(),
        );
    }
    said.push(rec.final_result().to_string());
    said
}

/// `[[rr:TD-13#Sample count defines time and delivery cadence]]`
/// `[[rr:TD-13#The host composes two independent stream lifecycles]]`
#[test]
fn a_host_composes_an_upstream_processor_with_the_recognizer() {
    let Some(dir) = model_dir() else { return };
    let m = Model::open(&dir).unwrap();
    // Two segments of one capture with a gap between them. The first ends mid-hop; the second
    // opens on a second of digital silence.
    let first = [clip("yes"), vec![0; 8000], clip("seven"), vec![0; 700]].concat();
    let gap = 8000;
    let second = [vec![0; 16000], clip("no"), vec![0; 333]].concat();
    let offset = first.len() + gap;
    for gather in [false, true] {
        let mut segments = Vec::new();
        for (start, capture) in [(0, &first), (offset, &second)] {
            // A discontinuity closes the segment: a fresh recognizer, the source offset kept.
            let mut host = Host::new(&m, gather);
            host.segment(capture);
            assert_eq!(host.calls.iter().sum::<usize>(), capture.len());
            let limit = if gather { BLOCK } else { HOP };
            assert!(host.calls.iter().all(|&n| n > 0 && n <= limit));
            assert!(*host.calls.last().unwrap() < limit, "{:?}", host.calls);
            assert_eq!(host.said, replay(&m, capture, &host.calls));
            segments.push((start, host.said));
        }
        let (start, said) = &segments[1];
        let fin = said
            .iter()
            .find(|s| s.contains("\"word\": \"no\""))
            .unwrap();
        let word = start + num_of_key(fin, "\"start_sample\": ").unwrap() as usize;
        let spoken = offset + 16000;
        assert!(
            word + 1600 >= spoken && word < spoken + clip("no").len(),
            "the word at source sample {word}, spoken from {spoken}"
        );
    }
}
