//! Gates G2, G3 and G5: stream WAV takes through the recognizer in fixed blocks, recording the
//! partial after every block, every final with its words and end sample, and compute time per
//! block, as JSON lines for `[[rr:scripts/g2.py#score]]` to compare with the stock wheel.
// [[rr:TD-2#Verification and acceptance]]

use std::path::PathBuf;
use std::time::Instant;
use utter::wav::read_wav;
use utter::{Model, Recognizer};

struct Run {
    block_ms: usize,
    alternatives: usize,
    partial_words: bool,
    opts: utter::recognizer::RecognizerOptions,
    endpoint_ms: Option<f32>,
    endpoint_veto: Option<f32>,
}

fn main() {
    let mut model_dir = PathBuf::new();
    let mut grammar_path = PathBuf::new();
    let mut wavs: Vec<PathBuf> = Vec::new();
    let mut block_ms = 40usize;
    let mut alternatives = 0usize;
    let mut partial_words = false;
    let mut dither: Option<f32> = None;
    let mut unknown_cost: Option<f32> = None;
    let mut silence_weight = utter::recognizer::SILENCE_WEIGHT;
    let mut endpoint_ms: Option<f32> = None;
    let mut endpoint_veto: Option<f32> = None;
    let mut threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model_dir = PathBuf::from(it.next().unwrap()),
            "--grammar" => grammar_path = PathBuf::from(it.next().unwrap()),
            "--block-ms" => block_ms = it.next().unwrap().parse().unwrap(),
            "--alternatives" => alternatives = it.next().unwrap().parse().unwrap(),
            "--partial-words" => partial_words = true,
            "--dither" => dither = Some(it.next().unwrap().parse().unwrap()),
            "--unknown-cost" => unknown_cost = Some(it.next().unwrap().parse().unwrap()),
            "--silence-weight" => silence_weight = it.next().unwrap().parse().unwrap(),
            "--endpoint-ms" => endpoint_ms = Some(it.next().unwrap().parse().unwrap()),
            "--endpoint-veto" => endpoint_veto = Some(it.next().unwrap().parse().unwrap()),
            "--threads" => threads = it.next().unwrap().parse().unwrap(),
            "--corpus" => {
                let dir = PathBuf::from(it.next().unwrap());
                let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
                    .expect("corpus dir")
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().map(|x| x == "wav").unwrap_or(false))
                    .collect();
                v.sort();
                wavs.extend(v);
            }
            other => wavs.push(PathBuf::from(other)),
        }
    }
    let t0 = Instant::now();
    let mut model = Model::open(&model_dir).expect("model");
    if let Some(d) = dither {
        model.mfcc_opts.dither = d;
    }
    eprintln!("model loaded in {:?}", t0.elapsed());
    let grammar =
        utter::json::parse_string_array(&std::fs::read_to_string(&grammar_path).expect("grammar"))
            .expect("grammar JSON");

    let run = Run {
        block_ms,
        alternatives,
        partial_words,
        endpoint_ms,
        endpoint_veto,
        opts: utter::recognizer::RecognizerOptions {
            unknown_cost,
            silence_weight,
            ..Default::default()
        },
    };

    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = std::sync::Mutex::new(Vec::<(usize, String)>::new());
    std::thread::scope(|sc| {
        for _ in 0..threads.min(wavs.len().max(1)) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= wavs.len() {
                    break;
                }
                let line = run_take(&model, &grammar, &wavs[i], &run);
                out.lock().unwrap().push((i, line));
            });
        }
    });
    let mut lines = out.into_inner().unwrap();
    lines.sort_by_key(|(i, _)| *i);
    for (_, l) in lines {
        println!("{l}");
    }
}

fn run_take(model: &Model, grammar: &[String], wav: &std::path::Path, run: &Run) -> String {
    let w = read_wav(wav).expect("wav");
    let t0 = Instant::now();
    let mut rec = Recognizer::with_options(model, w.sample_rate as f32, grammar, &run.opts)
        .expect("recognizer");
    rec.set_endpoint_bound(run.endpoint_ms, run.endpoint_veto);
    let t_new = t0.elapsed();
    rec.set_words(true);
    rec.set_partial_words(run.partial_words);
    rec.set_alternatives(run.alternatives);
    let block = w.sample_rate as usize * run.block_ms / 1000;
    let mut partials: Vec<String> = Vec::new();
    let mut segments: Vec<String> = Vec::new();
    let mut compute_us: Vec<u128> = Vec::new();
    let mut fed = 0usize;
    let mut i = 0;
    while i < w.samples.len() {
        let end = (i + block).min(w.samples.len());
        let t = Instant::now();
        let step = rec.accept(&w.samples[i..end]);
        fed = end;
        if step.endpoint {
            let res = rec.result().to_string();
            segments.push(format!(
                "{{\"end_sample\": {}, \"endpoint_sample\": {}, \"result\": {}}}",
                fed, step.sample, res
            ));
            partials.push(String::new());
        } else {
            partials.push(rec.partial().to_string());
        }
        compute_us.push(t.elapsed().as_micros());
        i = end;
    }
    let res = rec.final_result().to_string();
    segments.push(format!(
        "{{\"end_sample\": {}, \"endpoint_sample\": {}, \"result\": {}}}",
        fed, fed, res
    ));
    let mut s = String::from("{\"take\": ");
    utter::json::write_string(&mut s, wav.file_stem().unwrap().to_str().unwrap());
    s.push_str(&format!(", \"block_ms\": {}, \"samples\": {}, \"new_ms\": {}, \"graph_states\": {}, \"compute_us\": [", run.block_ms, w.samples.len(), t_new.as_millis(), rec.graph().num_states()));
    s.push_str(
        &compute_us
            .iter()
            .map(|u| u.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    s.push_str("], \"partials\": [");
    for (k, p) in partials.iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        if p.is_empty() {
            s.push_str("null");
        } else {
            s.push_str(p);
        }
    }
    s.push_str("], \"segments\": [");
    s.push_str(&segments.join(", "));
    s.push_str("]}");
    s
}
