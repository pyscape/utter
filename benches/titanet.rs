//! What one TitaNet embedding costs by span length, on one thread.
//!
//!     cargo bench --bench titanet -- --model DIR [--reps N]
//!
//! DIR is a directory scripts/titanet_convert.py wrote; without `--model` or
//! `UTTER_TEST_TITANET_MODEL` the bench is skipped. For each span length it reports the network
//! alone on features already computed, as the native prototype measured it, and the whole path
//! from samples: the minimum and the median over the repetitions, in milliseconds.

use std::hint::black_box;
use std::time::Instant;
use utter::fbank::NUM_BANDS;
use utter::titanet::{Embedding, TitaNet};

const FRAMES: &[usize] = &[25, 50, 100, 200];

fn summary(mut ms: Vec<f64>) -> (f64, f64) {
    ms.sort_by(f64::total_cmp);
    (ms[0], ms[ms.len() / 2])
}

// A span's sample count is far below the range an f32 holds exactly.
#[allow(clippy::cast_precision_loss)]
fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let arg = |name: &str| {
        argv.iter()
            .position(|a| a == name)
            .and_then(|i| argv.get(i + 1).cloned())
    };
    let reps: usize = arg("--reps").map_or(30, |r| r.parse().expect("--reps N"));
    let Some(dir) = arg("--model").or_else(|| std::env::var("UTTER_TEST_TITANET_MODEL").ok())
    else {
        eprintln!("titanet: skipped, no --model and no UTTER_TEST_TITANET_MODEL");
        return;
    };
    let model = TitaNet::open(std::path::Path::new(&dir)).expect("TitaNet model");
    let mut seed = 1u64;
    let mut noise = move || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let mut job = Embedding::new();
    println!("frames  network min / median ms  whole path min / median ms");
    for &t in FRAMES {
        let features: Vec<f32> = (0..t * NUM_BANDS).map(|_| noise()).collect();
        let samples: Vec<f32> = (0..400 + (t - 1) * 160).map(|_| 0.1 * noise()).collect();
        let mut time = |whole: bool| -> Vec<f64> {
            (0..=reps)
                .map(|_| {
                    let start = Instant::now();
                    if whole {
                        job.begin(&model, &samples).unwrap();
                    } else {
                        job.begin_features(&model, &features).unwrap();
                    }
                    job.advance(&model, u64::MAX);
                    black_box(job.take());
                    start.elapsed().as_secs_f64() * 1e3
                })
                .skip(1)
                .collect()
        };
        let (net_min, net_med) = summary(time(false));
        let (all_min, all_med) = summary(time(true));
        println!("{t:6}  {net_min:10.3} / {net_med:.3}  {all_min:16.3} / {all_med:.3}");
    }
}
