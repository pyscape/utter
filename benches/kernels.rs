//! The two costs the decision records quote, in a form anyone can re-run: the matrix kernel at
//! the network's real shapes and grammar composition against grammar size.
//!
//!     cargo bench --bench kernels
//!     cargo bench --bench kernels -- --reps 40 --model /path/to/vosk-model-small-en-us-0.15
//!
//! The grammar half needs a model and is skipped unless `--model` or `UTTER_TEST_MODEL` names
//! one. `[[rr:TD-2#Dependency policy]]` rules out a benchmark harness, so this is a plain binary
//! behind `harness = false`.
//!
//! The figure reported is the minimum over `--reps`. The quantity wanted is what the kernel costs
//! when nothing interrupts it; a mean over a shared machine measures the rest of the machine as
//! much as this, and the minimum is the only summary that does not move when a neighbour wakes up.
//! The spread against the median is printed beside it, because a minimum alone cannot say whether
//! the run was quiet enough to believe.
use std::hint::black_box;
use std::time::Instant;
use utter::gemm::gemm_abt;
use utter::Model;

/// `(k, n, how many times the inference path runs it)`, read off the stock
/// `vosk-model-small-en-us-0.15` weight matrices. Rows are the chunk's frames and come from
/// `--rows`. The chain branch only: `output-xent` and `prefinal-xent` are training-time nodes
/// and are never evaluated for a decode. `[[rr:TD-3]]` quotes totals over these.
const SHAPES: &[(usize, usize, usize)] = &[
    (1280, 96, 10), // tdnnf{2,3,4,6..12}.linear
    (192, 640, 7),  // tdnnf{2,3,4,6,7,8}.affine, prefinal-chain.affine
    (96, 640, 5),   // tdnnf{5,9,10,11,12}.affine
    (640, 192, 2),  // prefinal-l, prefinal-chain.linear
    (640, 96, 1),   // tdnnf5.linear
    (192, 2192, 1), // output.affine
    (150, 640, 1),  // tdnn1.affine
];

/// Grammar sizes for the composition sweep, the sizes the public benchmark's own sweep reports.
const GRAMMAR_SIZES: &[usize] = &[35, 60, 92, 150, 200, 246];

struct Args {
    reps: usize,
    rows: usize,
    model: Option<String>,
    filter: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        reps: 20,
        rows: 24,
        model: std::env::var("UTTER_TEST_MODEL").ok(),
        filter: None,
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
            "--rows" => {
                a.rows = need(i)?.parse().map_err(|e| format!("--rows: {e}"))?;
                i += 2;
            }
            "--model" => {
                a.model = Some(need(i)?);
                i += 2;
            }
            "--filter" => {
                a.filter = Some(need(i)?);
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.reps == 0 || a.rows == 0 {
        return Err("--reps and --rows must be positive".into());
    }
    Ok(a)
}

/// Minimum and median seconds over `reps` calls of `f`.
fn time<F: FnMut()>(reps: usize, mut f: F) -> (f64, f64) {
    let mut t: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        f();
        t.push(start.elapsed().as_secs_f64());
    }
    t.sort_by(|x, y| x.partial_cmp(y).unwrap());
    (t[0], t[t.len() / 2])
}

/// Deterministic operands. The kernel's cost does not depend on the values, but reusing one
/// buffer across shapes would let a shape inherit the previous one's cache state.
fn operand(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i + seed) * 2654435761) % 65521) as f32 / 32760.0) - 1.0)
        .collect()
}

fn bench_gemm(args: &Args) {
    println!(
        "gemm_abt, {} rows per call, minimum of {}\n",
        args.rows, args.reps
    );
    println!("| shape m x k x n | per chunk | min ms | median ms | GFLOP/s | chunk ms |");
    println!("|---|---|---|---|---|---|");
    let m = args.rows;
    let mut chunk_total = 0.0;
    for &(k, n, count) in SHAPES {
        let a = operand(m * k, 1);
        let b = operand(n * k, 2);
        let mut c = vec![0.0f32; m * n];
        let (min, med) = time(args.reps, || {
            gemm_abt(black_box(&a), m, k, black_box(&b), n, None, &mut c);
            black_box(c[0]);
        });
        let flops = 2.0 * (m * k * n) as f64;
        chunk_total += min * count as f64;
        println!(
            "| {m}x{k}x{n} | {count} | {:.4} | {:.4} | {:.1} | {:.3} |",
            min * 1e3,
            med * 1e3,
            flops / min / 1e9,
            min * count as f64 * 1e3,
        );
    }
    println!(
        "\nSummed over the inference path at {m} rows for every layer: {:.3} ms. That is the whole \
         path's matrix product, not a measured chunk: a real chunk subsamples, so the layers past \
         the subsampling point run fewer rows than the ones before it. Pass --rows to move it.",
        chunk_total * 1e3
    );
}

fn bench_compile_grammar(args: &Args, dir: &str) -> Result<(), String> {
    let model = Model::open(std::path::Path::new(dir)).map_err(|e| format!("{dir}: {e:?}"))?;
    // The vocabulary in a fixed order: a HashMap's own order is seeded per process, and a sweep
    // whose grammar changed between runs would measure a different grammar, not a faster one.
    let mut vocab: Vec<&str> = model
        .word_ids
        .keys()
        .map(|s| s.as_str())
        .filter(|w| !w.starts_with('[') && !w.starts_with('<') && !w.contains('#') && *w != "<eps>")
        .collect();
    vocab.sort_unstable();
    println!(
        "\ncompile_grammar, {}, minimum of {}\n",
        model.dir.display(),
        args.reps
    );
    println!("| entries | min ms | median ms | states |");
    println!("|---|---|---|---|");
    for &size in GRAMMAR_SIZES {
        if size > vocab.len() {
            break;
        }
        let grammar: Vec<String> = vocab[..size].iter().map(|s| s.to_string()).collect();
        // compile_grammar is the uncompiled path; the cache `[[rr:TD-4]]` keeps sits above it in
        // grammar_graph, so repeating this measures compilation and not a lookup.
        let mut states = 0;
        let (min, med) = time(args.reps, || {
            let g = model
                .compile_grammar(black_box(&grammar), None, 5_000_000, |_| {})
                .expect("compile");
            states = g.states.len();
            black_box(states);
        });
        println!(
            "| {size} | {:.2} | {:.2} | {states} |",
            min * 1e3,
            med * 1e3
        );
    }
    Ok(())
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\nusage: cargo bench --bench kernels -- [--reps N] [--rows N] [--model DIR] [--filter gemm|grammar]");
            std::process::exit(2);
        }
    };
    let want = |name: &str| args.filter.as_deref().is_none_or(|f| name.contains(f));
    if want("gemm") {
        bench_gemm(&args);
    }
    if want("grammar") {
        match args.model.as_deref() {
            Some(dir) => {
                if let Err(e) = bench_compile_grammar(&args, dir) {
                    eprintln!("grammar: {e}");
                    std::process::exit(1);
                }
            }
            None => eprintln!("\ngrammar: skipped, no --model and no UTTER_TEST_MODEL"),
        }
    }
}
