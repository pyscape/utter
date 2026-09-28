//! Normalized float samples lent to `accept_f32` in 10, 20 and 40 ms calls, beside the
//! established baseline of 16-bit samples to `accept` in 40 ms calls, over one stream of words
//! and pauses: compute per call, allocations per call, the heap the recognizer holds as the
//! stream grows, and where in the audio each result arrives.
//!
//!     cargo bench --bench float_delivery -- --model /path/to/vosk-model-small-en-us-0.15
//!     cargo bench --bench float_delivery -- --reps 9 --seconds 120
//!
//! Skipped unless `--model` or `UTTER_TEST_MODEL` names a model. `[[rr:TD-2#Dependency policy]]`
//! rules out a harness and an allocation profiler, so this is a plain binary behind
//! `harness = false` that counts through its own global allocator.
//!
//! Compute is the minimum over `--reps` for each call, as in the kernels bench, and the host
//! reads the partial or the final after every call, as a host acting on partials does; the two
//! are timed apart. The stream is the three clips under `tests/data` between pauses of digital
//! silence and of hiss, repeated to `--seconds`.
// [[rr:TD-13#Sample count defines time and delivery cadence]]

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;
use utter::{Model, Recognizer};

struct Counting;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED: AtomicU64 = AtomicU64::new(0);
static FREED: AtomicU64 = AtomicU64::new(0);

// SAFETY: every call is passed to the system allocator unchanged; the counters are atomics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Relaxed);
        ALLOCATED.fetch_add(layout.size() as u64, Relaxed);
        // SAFETY: the caller's contract for `alloc` is the system allocator's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Relaxed);
        ALLOCATED.fetch_add(layout.size() as u64, Relaxed);
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Relaxed);
        ALLOCATED.fetch_add(new_size as u64, Relaxed);
        FREED.fetch_add(layout.size() as u64, Relaxed);
        // SAFETY: `ptr` came from this allocator, which is the system's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        FREED.fetch_add(layout.size() as u64, Relaxed);
        // SAFETY: as for `realloc`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

#[allow(clippy::cast_possible_wrap)]
fn live() -> i64 {
    (ALLOCATED.load(Relaxed) - FREED.load(Relaxed)) as i64
}

const RATE: usize = 16000;

struct Args {
    reps: usize,
    seconds: usize,
    model: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        reps: 5,
        seconds: 60,
        model: std::env::var("UTTER_TEST_MODEL").ok(),
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let need = |i: usize| -> Result<String, String> {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} wants a value", argv[i]))
        };
        match argv[i].as_str() {
            // `cargo bench` passes this through to a harness-less binary; it is not ours.
            "--bench" => i += 1,
            "--reps" => {
                a.reps = need(i)?.parse().map_err(|e| format!("--reps: {e}"))?;
                i += 2;
            }
            "--seconds" => {
                a.seconds = need(i)?.parse().map_err(|e| format!("--seconds: {e}"))?;
                i += 2;
            }
            "--model" => {
                a.model = Some(need(i)?);
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.reps == 0 || a.seconds == 0 {
        return Err("--reps and --seconds must be positive".into());
    }
    Ok(a)
}

fn clip(name: &str) -> Vec<i16> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(format!("{name}.wav"));
    utter::wav::read_wav(&path).expect("test clip").samples
}

/// Uniform noise at 50 dB below full scale, from a fixed seed.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn hiss(n: usize, seed: u64) -> Vec<i16> {
    let peak = 32768.0 * 10f64.powf(-50.0 / 20.0) * 3f64.sqrt();
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as f64 / f64::from(1u32 << 31);
            ((u * 2.0 - 1.0) * peak).round() as i16
        })
        .collect()
}

fn stream(seconds: usize) -> Vec<i16> {
    let (yes, seven, no) = (clip("yes"), clip("seven"), clip("no"));
    let mut out = Vec::with_capacity(seconds * RATE);
    let mut seed = 1;
    while out.len() < seconds * RATE {
        out.extend(&yes);
        out.extend(hiss(RATE, seed));
        out.extend(&seven);
        out.extend(vec![0; RATE * 3 / 2]);
        out.extend(&no);
        out.extend(hiss(RATE * 2, seed + 1));
        seed += 2;
    }
    out.truncate(seconds * RATE);
    out
}

#[derive(Clone, Copy, PartialEq)]
struct Delivery {
    ms: usize,
    float: bool,
}

impl Delivery {
    fn name(self) -> String {
        format!("{} {} ms", if self.float { "f32" } else { "i16" }, self.ms)
    }
}

/// One call and what the host read after it.
struct Call {
    fed: u64,
    accept_ns: u64,
    read_ns: u64,
    accept_allocations: u64,
    read_allocations: u64,
    endpoint: bool,
    decoded: u64,
    words: Vec<String>,
}

struct Run {
    calls: Vec<Call>,
    finals: Vec<String>,
    /// Heap the calls left allocated, not counting the bench's own records of them, at the
    /// stream's half and at its end.
    held: (i64, i64),
}

fn words_of(json: &str, key: &str) -> Vec<String> {
    let Some(i) = json.find(key) else {
        return Vec::new();
    };
    let rest = &json[i + key.len()..];
    rest[..rest.find('"').unwrap_or(0)]
        .split(' ')
        .filter(|w| !w.is_empty() && !w.starts_with('['))
        .map(str::to_string)
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
fn run(model: &Model, grammar: &[String], pcm: &[i16], float: &[f32], d: Delivery) -> Run {
    let mut rec = Recognizer::new(model, 16_000.0, grammar).expect("recognizer");
    rec.set_words(true);
    rec.set_partial_words(true);
    rec.set_alternatives(3);
    let n = RATE / 1000 * d.ms;
    let (mut calls, mut finals) = (Vec::new(), Vec::new());
    let (mut fed, mut held, mut half) = (0, 0, 0);
    for start in (0..pcm.len()).step_by(n) {
        if start <= pcm.len() / 2 {
            half = held;
        }
        let end = (start + n).min(pcm.len());
        let l0 = live();
        let a0 = ALLOCATIONS.load(Relaxed);
        let t = Instant::now();
        let step = if d.float {
            rec.accept_f32(&float[start..end]).expect("normalized")
        } else {
            rec.accept(&pcm[start..end])
        };
        let accept_ns = t.elapsed().as_nanos() as u64;
        let a1 = ALLOCATIONS.load(Relaxed);
        let t = Instant::now();
        let json = if step.endpoint {
            rec.result()
        } else {
            rec.partial()
        };
        let read_ns = t.elapsed().as_nanos() as u64;
        let (accept_allocations, read_allocations) = (a1 - a0, ALLOCATIONS.load(Relaxed) - a1);
        held += live() - l0;
        fed += (end - start) as u64;
        let words = if step.endpoint {
            finals.push(words_of(json, "\"text\": \"").join(" "));
            Vec::new()
        } else {
            words_of(json, "\"partial\": \"")
        };
        calls.push(Call {
            fed,
            accept_ns,
            read_ns,
            accept_allocations,
            read_allocations,
            endpoint: step.endpoint,
            decoded: step.sample,
            words,
        });
    }
    rec.final_result();
    Run {
        calls,
        finals,
        held: (half, held),
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn percentile(sorted: &[u64], p: f64) -> u64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// Where in the audio fed each word first stood in a partial, keyed by the finals before it,
/// its place in the partial and the word.
fn first_sightings(r: &Run) -> HashMap<(usize, usize, String), u64> {
    let mut out = HashMap::new();
    let mut utterance = 0;
    for c in &r.calls {
        if c.endpoint {
            utterance += 1;
            continue;
        }
        for (k, w) in c.words.iter().enumerate() {
            out.entry((utterance, k, w.clone())).or_insert(c.fed);
        }
    }
    out
}

#[allow(clippy::cast_precision_loss)]
fn ms(samples: f64) -> f64 {
    samples * 1000.0 / RATE as f64
}

#[allow(clippy::cast_precision_loss)]
fn report(model: &Model, args: &Args) {
    let grammar: Vec<String> = [
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "one", "two",
        "three", "four", "five", "six", "seven", "eight", "nine", "zero",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let pcm = stream(args.seconds);
    let float: Vec<f32> = pcm.iter().map(|&s| f32::from(s) / 32768.0).collect();
    let deliveries = [
        Delivery {
            ms: 40,
            float: false,
        },
        Delivery {
            ms: 10,
            float: true,
        },
        Delivery {
            ms: 20,
            float: true,
        },
        Delivery {
            ms: 40,
            float: true,
        },
    ];
    // One untimed run first, so the grammar's graph is composed and cached before any count.
    run(model, &grammar, &pcm, &float, deliveries[0]);
    // The deliveries take turns within each rep, so a neighbour busy for a while slows them
    // alike rather than whichever ran then.
    let mut runs: Vec<Run> = deliveries
        .iter()
        .map(|&d| run(model, &grammar, &pcm, &float, d))
        .collect();
    for _ in 1..args.reps {
        for (&d, best) in deliveries.iter().zip(runs.iter_mut()) {
            let r = run(model, &grammar, &pcm, &float, d);
            for (x, y) in best.calls.iter_mut().zip(&r.calls) {
                x.accept_ns = x.accept_ns.min(y.accept_ns);
                x.read_ns = x.read_ns.min(y.read_ns);
            }
        }
    }
    let seconds = pcm.len() as f64 / RATE as f64;
    println!(
        "{} s of audio, {}, minimum of {} per call\n",
        seconds,
        model.dir.display(),
        args.reps
    );
    println!(
        "| delivery | calls | accept p50 us | p95 us | p99 us | max us | read p50 us | ms per s of audio |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for (d, r) in deliveries.iter().zip(&runs) {
        let mut accept: Vec<u64> = r.calls.iter().map(|c| c.accept_ns).collect();
        accept.sort_unstable();
        let mut read: Vec<u64> = r.calls.iter().map(|c| c.read_ns).collect();
        read.sort_unstable();
        let total: u64 = r.calls.iter().map(|c| c.accept_ns + c.read_ns).sum();
        println!(
            "| {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} |",
            d.name(),
            r.calls.len(),
            percentile(&accept, 0.5) as f64 / 1e3,
            percentile(&accept, 0.95) as f64 / 1e3,
            percentile(&accept, 0.99) as f64 / 1e3,
            accept[accept.len() - 1] as f64 / 1e3,
            percentile(&read, 0.5) as f64 / 1e3,
            total as f64 / 1e6 / seconds,
        );
    }
    println!(
        "\n| delivery | allocations per accept, mean | max | per read, mean | both, per s of audio | heap held after the stream, MB | growth per s over its second half, KB |"
    );
    println!("|---|---|---|---|---|---|---|");
    for (d, r) in deliveries.iter().zip(&runs) {
        let accept: u64 = r.calls.iter().map(|c| c.accept_allocations).sum();
        let max = r.calls.iter().map(|c| c.accept_allocations).max();
        let read: u64 = r.calls.iter().map(|c| c.read_allocations).sum();
        let calls = r.calls.len() as f64;
        println!(
            "| {} | {:.1} | {} | {:.1} | {:.0} | {:.2} | {:.1} |",
            d.name(),
            accept as f64 / calls,
            max.unwrap_or(0),
            read as f64 / calls,
            (accept + read) as f64 / seconds,
            r.held.1 as f64 / 1e6,
            (r.held.1 - r.held.0) as f64 / 1e3 / (seconds / 2.0),
        );
    }
    let base = &runs[0];
    let base_seen = first_sightings(base);
    let base_ends: Vec<u64> = base
        .calls
        .iter()
        .filter(|c| c.endpoint)
        .map(|c| c.fed)
        .collect();
    println!(
        "\nArrival is the audio fed when the result was read, against i16 40 ms; negative is earlier."
    );
    println!(
        "\n| delivery | partial changes | finals, same as i16 40 ms | word first seen, mean ms | min | max | final, mean ms | min | max | fed less decoded p50 ms | max |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for (d, r) in deliveries.iter().zip(&runs) {
        let changes = r
            .calls
            .windows(2)
            .filter(|w| w[0].words != w[1].words)
            .count();
        let seen = first_sightings(r);
        let diffs: Vec<f64> = seen
            .iter()
            .filter_map(|(k, &at)| base_seen.get(k).map(|&b| at as f64 - b as f64))
            .collect();
        let ends: Vec<u64> = r
            .calls
            .iter()
            .filter(|c| c.endpoint)
            .map(|c| c.fed)
            .collect();
        let same_finals = r.finals == base.finals;
        let end_diffs: Vec<f64> = if same_finals {
            ends.iter()
                .zip(&base_ends)
                .map(|(&a, &b)| a as f64 - b as f64)
                .collect()
        } else {
            Vec::new()
        };
        let mut lag: Vec<u64> = r.calls.iter().map(|c| c.fed - c.decoded).collect();
        lag.sort_unstable();
        let spread = |v: &[f64]| {
            if v.is_empty() {
                return "- | - | -".to_string();
            }
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let min = v.iter().copied().fold(f64::INFINITY, f64::min);
            let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            format!("{:.1} | {:.0} | {:.0}", ms(mean), ms(min), ms(max))
        };
        println!(
            "| {} | {} | {}, {} | {} | {} | {:.0} | {:.0} |",
            d.name(),
            changes,
            r.finals.len(),
            if same_finals { "same" } else { "differ" },
            spread(&diffs),
            spread(&end_diffs),
            ms(percentile(&lag, 0.5) as f64),
            ms(lag[lag.len() - 1] as f64),
        );
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!(
                "{e}\nusage: cargo bench --bench float_delivery -- [--reps N] [--seconds N] [--model DIR]"
            );
            std::process::exit(2);
        }
    };
    let Some(dir) = args.model.as_deref() else {
        eprintln!("float_delivery: skipped, no --model and no UTTER_TEST_MODEL");
        return;
    };
    let model = Model::open(Path::new(dir)).unwrap_or_else(|e| {
        eprintln!("{dir}: {e}");
        std::process::exit(1);
    });
    report(&model, &args);
}
