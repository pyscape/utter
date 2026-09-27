//! Tests against TitaNet-small converted by scripts/titanet_convert.py: skipped unless
//! `UTTER_TEST_TITANET_MODEL` names the directory it wrote. The references in
//! tests/data/titanet-small.txt are written by `scripts/titanet_oracle.py --test-vectors`.

use std::path::PathBuf;
use utter::titanet::{Embedding, TitaNet};
use utter::wav::read_wav;

fn model() -> Option<TitaNet> {
    let dir = std::env::var_os("UTTER_TEST_TITANET_MODEL").map(PathBuf::from)?;
    dir.join("titanet.conf")
        .exists()
        .then(|| TitaNet::open(&dir).unwrap())
}

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn clip(name: &str) -> Vec<f32> {
    read_wav(&data().join(format!("{name}.wav")))
        .unwrap()
        .samples
        .iter()
        .map(|&s| f32::from(s) / 32768.0)
        .collect()
}

struct Reference {
    name: String,
    rate: u32,
    start: usize,
    end: usize,
    vector: Vec<f32>,
}

fn references() -> Vec<Reference> {
    std::fs::read_to_string(data().join("titanet-small.txt"))
        .unwrap()
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.split(' ').collect();
            Reference {
                name: f[0].to_string(),
                rate: f[1].parse().unwrap(),
                start: f[2].parse().unwrap(),
                end: f[3].parse().unwrap(),
                vector: f[4..].iter().map(|v| v.parse().unwrap()).collect(),
            }
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / norm(a) / norm(b)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn embeddings_match_the_references() {
    let Some(m) = model() else {
        return;
    };
    assert_eq!(m.dim(), 192);
    for r in references() {
        let x = clip(&r.name);
        let got = if r.rate == 16_000 {
            m.embed(&x[r.start..r.end]).unwrap()
        } else {
            let held: Vec<f32> = x.iter().flat_map(|&s| [s; 3]).collect();
            m.embed_at_rate(&held[r.start..r.end], r.rate as f32)
                .unwrap()
        };
        let (c, d) = (cosine(&got, &r.vector), max_abs(&got, &r.vector));
        assert!(
            c >= 0.999_999_8 && d <= 1e-4,
            "{} at {} Hz, {}..{}: cosine {c}, max abs {d}",
            r.name,
            r.rate,
            r.start,
            r.end
        );
    }
}

/// A small generator, so the budgets are the same on every run.
fn budgets(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        1 + s % 3_000_000
    }
}

#[test]
fn slices_give_the_bytes_of_one_call() {
    let Some(m) = model() else {
        return;
    };
    let mut job = Embedding::new();
    for (i, name) in ["yes", "no", "seven"].iter().enumerate() {
        let x = clip(name);
        for span in [&x[..], &x[2000..2000 + 400 + 30 * 160]] {
            let whole = m.embed(span).unwrap();
            let mut next = budgets(0x9e37_79b9 + i as u64);
            // Random budgets, one frame of a stage per call, and budgets per call like a
            // recognizer's blocks of 10, 20, 40 and 200 ms at a fixed share of their audio.
            let schedules: [&mut dyn FnMut() -> u64; 6] = [
                &mut next,
                &mut || 1,
                &mut || 10 * 40_000,
                &mut || 20 * 40_000,
                &mut || 40 * 40_000,
                &mut || 200 * 40_000,
            ];
            for budget in schedules {
                job.begin(&m, span).unwrap();
                let mut calls = 0;
                while !job.is_done() {
                    job.advance(&m, budget());
                    calls += 1;
                }
                assert!(calls > 1);
                let got = job.take();
                let same = got
                    .iter()
                    .zip(&whole)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same && got.len() == whole.len(), "{name}");
            }
        }
    }
}

#[test]
fn spans_it_cannot_embed_are_refused() {
    let Some(m) = model() else {
        return;
    };
    let x = clip("yes");
    assert!(m.embed(&x[..399]).is_err());
    assert!(m.embed(&x[..400]).is_ok());
    assert!(m.embed_at_rate(&x, 8000.0).is_err());
    assert!(m.embed_at_rate(&x, 44_100.5).is_err());
    let long = vec![0.0f32; 400 + utter::titanet::MAX_FRAMES * 160];
    assert!(m.embed(&long).is_err());
    assert!(m.embed(&long[160..]).is_ok());
}
