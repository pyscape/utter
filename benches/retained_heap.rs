//! What a recognizer keeps on the heap as a stream goes on: the live bytes the process holds,
//! less those it held before the stream's first sample, sampled every ten seconds of audio.
//!
//!     cargo bench --bench retained_heap -- --model /path/to/vosk-model-small-en-us-0.15
//!     cargo bench --bench retained_heap -- --seconds 1200 --readings 3 --alternatives 10
//!     cargo bench --bench retained_heap -- --spk /path/to/vosk-model-spk-0.4 --margin 8
//!
//! The stream is the repository's test clips in turn with half a second of silence between,
//! fed in 40 ms blocks. A partial is read after every block and a result on every endpoint, as a
//! host reading the stream would, and the stream is closed with a result. The figures are what
//! a counting allocator saw, so they do not depend on the platform allocator's own bookkeeping.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use utter::{Model, Recognizer, SpeakerModel};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(new_size, Ordering::Relaxed);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

struct Args {
    model: Option<String>,
    spk: Option<String>,
    seconds: Vec<usize>,
    readings: usize,
    alternatives: usize,
    margin: Option<f32>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        model: std::env::var("UTTER_TEST_MODEL").ok(),
        spk: None,
        seconds: vec![60, 1200],
        readings: 0,
        alternatives: 0,
        margin: None,
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
            "--model" => {
                a.model = Some(need(i)?);
                i += 2;
            }
            "--spk" => {
                a.spk = Some(need(i)?);
                i += 2;
            }
            "--seconds" => {
                a.seconds = need(i)?
                    .split(',')
                    .map(|s| s.parse().map_err(|e| format!("--seconds: {e}")))
                    .collect::<Result<_, _>>()?;
                i += 2;
            }
            "--readings" => {
                a.readings = need(i)?.parse().map_err(|e| format!("--readings: {e}"))?;
                i += 2;
            }
            "--alternatives" => {
                a.alternatives = need(i)?
                    .parse()
                    .map_err(|e| format!("--alternatives: {e}"))?;
                i += 2;
            }
            "--margin" => {
                a.margin = Some(need(i)?.parse().map_err(|e| format!("--margin: {e}"))?);
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

fn clips() -> Vec<i16> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut out = Vec::new();
    for name in ["yes", "seven", "no"] {
        let wav = utter::wav::read_wav(&dir.join(format!("{name}.wav"))).expect("test clip");
        out.extend_from_slice(&wav.samples);
        out.extend(std::iter::repeat_n(0i16, 8000));
    }
    out
}

#[allow(clippy::cast_precision_loss)]
fn kib(bytes: isize) -> f64 {
    bytes as f64 / 1024.0
}

#[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
fn run(model: &Model, spk: Option<&SpeakerModel>, args: &Args, seconds: usize) {
    const RATE: usize = 16000;
    const BLOCK: usize = 640;
    let audio = clips();
    let grammar: Vec<String> = ["yes", "no", "seven", "[unk]"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut rec = Recognizer::new(model, RATE as f32, &grammar).expect("recognizer");
    rec.set_words(true);
    rec.set_partial_words(true);
    rec.set_alternatives(args.readings);
    rec.set_max_alternatives(args.alternatives);
    rec.set_endpoint_floor_margin(args.margin);
    if spk.is_some() {
        rec.set_spk_model(spk).expect("speaker model");
    }
    let base = LIVE.load(Ordering::Relaxed) as isize;
    let calls_before = CALLS.load(Ordering::Relaxed);
    let total = seconds * RATE;
    let mut fed = 0usize;
    let mut block = Vec::with_capacity(BLOCK);
    let mut samples: Vec<(usize, isize)> = Vec::new();
    let mut blocks = 0u64;
    while fed < total {
        block.clear();
        while block.len() < BLOCK {
            block.push(audio[(fed + block.len()) % audio.len()]);
        }
        let step = rec.accept(&block);
        if step.endpoint {
            std::hint::black_box(rec.result());
        } else {
            std::hint::black_box(rec.partial());
        }
        fed += BLOCK;
        blocks += 1;
        if fed.is_multiple_of(10 * RATE) {
            samples.push((fed / RATE, LIVE.load(Ordering::Relaxed) as isize - base));
        }
    }
    std::hint::black_box(rec.result());
    let end = LIVE.load(Ordering::Relaxed) as isize - base;
    let calls = CALLS.load(Ordering::Relaxed) - calls_before;
    // The slope over the first minute, and the largest the stream held at any sample.
    let (first_t, first_b) = samples.first().copied().unwrap_or((0, 0));
    let (min_t, min_b) = samples
        .iter()
        .copied()
        .take_while(|&(t, _)| t <= 60)
        .last()
        .unwrap_or((first_t, first_b));
    let slope = if min_t > first_t {
        (min_b - first_b) as f64 / (min_t - first_t) as f64
    } else {
        0.0
    };
    let peak = samples.iter().map(|s| s.1).max().unwrap_or(0).max(end);
    println!(
        "{seconds:>5} s: growth {:>7.1} KiB/s over the first minute, {:>9.1} KiB at the end, \
         {:>9.1} KiB at the peak, {:.1} allocations per call",
        slope / 1024.0,
        kib(end),
        kib(peak),
        calls as f64 / blocks as f64
    );
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("retained_heap: {e}");
            std::process::exit(2);
        }
    };
    let Some(path) = args.model.as_deref() else {
        println!("retained_heap: no model; set UTTER_TEST_MODEL or pass --model");
        return;
    };
    let model = Model::open(std::path::Path::new(path)).expect("model");
    let spk = args
        .spk
        .as_deref()
        .map(|p| SpeakerModel::open(std::path::Path::new(p)).expect("speaker model"));
    println!(
        "readings {}, alternatives {}, margin {:?}, speaker model {}",
        args.readings,
        args.alternatives,
        args.margin,
        if spk.is_some() { "x-vector" } else { "none" }
    );
    for &s in &args.seconds {
        run(&model, spk.as_ref(), &args, s);
    }
}
