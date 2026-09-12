//! Gate G0: batch-decode a corpus of WAV takes through the vendored acoustic and graph code with
//! the grammar composed offline, in zero and faithful i-vector modes, emitting one JSON line per
//! take for scripts/g0.py to score against libvosk's finals.
// [[rr:TD-2#Verification and acceptance]]

use std::path::{Path, PathBuf};
use std::time::Instant;
use utter::decode::BatchDecoder;
use utter::ivector::IvectorStream;
use utter::mfcc::Mfcc;
use utter::nnet3::Mat;
use utter::wav::read_wav;
use utter::Model;

struct Args {
    model: PathBuf,
    grammar: PathBuf,
    wavs: Vec<PathBuf>,
    modes: Vec<String>,
    dither: Option<f32>,
    step_ms: usize,
    threads: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: PathBuf::new(),
        grammar: PathBuf::new(),
        wavs: Vec::new(),
        modes: vec!["zero".into(), "faithful".into()],
        dither: None,
        step_ms: 40,
        threads: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => a.model = PathBuf::from(it.next().expect("--model DIR")),
            "--grammar" => a.grammar = PathBuf::from(it.next().expect("--grammar FILE")),
            "--modes" => {
                a.modes = it
                    .next()
                    .expect("--modes a,b")
                    .split(',')
                    .map(String::from)
                    .collect()
            }
            "--dither" => a.dither = Some(it.next().expect("--dither F").parse().unwrap()),
            "--step-ms" => a.step_ms = it.next().expect("--step-ms N").parse().unwrap(),
            "--threads" => a.threads = it.next().expect("--threads N").parse().unwrap(),
            "--corpus" => {
                let dir = PathBuf::from(it.next().expect("--corpus DIR"));
                let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
                    .expect("corpus dir")
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().map(|x| x == "wav").unwrap_or(false))
                    .collect();
                v.sort();
                a.wavs.extend(v);
            }
            other => a.wavs.push(PathBuf::from(other)),
        }
    }
    a
}

/// MFCC frames complete after `n` samples fed.
fn frames_ready(n: usize, mfcc: &Mfcc) -> usize {
    mfcc.num_frames(n)
}

/// Run the network over output frames in chunks, each chunk with one i-vector chosen by `ivec`.
fn forward_chunked(
    model: &Model,
    feats: &Mat,
    chunk_in: usize,
    mut ivec: impl FnMut(usize, usize) -> Vec<f32>,
) -> Mat {
    let net = &model.net;
    let sf = model.conf.frame_subsampling_factor;
    let t = feats.r;
    let n_out = t.div_ceil(sf);
    let (lctx, rctx) = net.context;
    let mut out = Mat::new(n_out, net.output_dim);
    let mut k = 0;
    while k * chunk_in < t {
        let begin = k * chunk_in;
        let end = ((k + 1) * chunk_in).min(t);
        let iv = ivec(begin, end);
        let win_begin = begin as i64 - lctx as i64;
        let win_end = end as i64 + rctx as i64;
        let win_len = (win_end - win_begin) as usize;
        let mut window = Mat::new(win_len, feats.c);
        for (i, ti) in (win_begin..win_end).enumerate() {
            let src = ti.clamp(0, t as i64 - 1) as usize;
            window.d[i * feats.c..(i + 1) * feats.c].copy_from_slice(feats.row(src));
        }
        let y = net.forward_window(window, &iv);
        let mut ti = begin;
        while ti < end {
            if ti.is_multiple_of(sf) {
                let o = ti / sf;
                let row = ti - begin + lctx;
                out.d[o * out.c..(o + 1) * out.c].copy_from_slice(y.row(row));
            }
            ti += 1;
        }
        k += 1;
    }
    out
}

fn decode_take(
    model: &Model,
    graph: &utter::fst::VectorFst,
    wav: &Path,
    mode: &str,
    dither: Option<f32>,
    step_ms: usize,
) -> String {
    let w = read_wav(wav).expect("wav");
    assert_eq!(w.channels, 1, "mono only");
    let samples: Vec<f32> = w.samples.iter().map(|&s| s as f32).collect();
    let t0 = Instant::now();
    let mut opts = model.mfcc_opts.clone();
    if let Some(d) = dither {
        opts.dither = d;
    }
    let mut mfcc = Mfcc::new(&opts);
    let feats = mfcc.compute(&samples);
    let t_feat = t0.elapsed();
    let sf = model.conf.frame_subsampling_factor;
    let chunk_in = model.conf.frames_per_chunk;
    let ivdim = model.net.ivector_dim;
    let t1 = Instant::now();
    let loglikes = match mode {
        "zero" => forward_chunked(model, &feats, chunk_in, |_, _| vec![0.0; ivdim]),
        "faithful" => {
            let info = model
                .ivector
                .as_ref()
                .expect("model has no i-vector extractor");
            let mut stream = IvectorStream::new(info);
            let rctx = model.net.context.1;
            let total = feats.r;
            let step = w.sample_rate as usize * step_ms / 1000;
            let n_samples = samples.len();
            // frames complete after each accept step; the last step carries the remainder
            let mut steps: Vec<usize> = Vec::new();
            let mut fed = 0;
            while fed < n_samples {
                fed = (fed + step).min(n_samples);
                steps.push(frames_ready(fed, &mfcc));
            }
            let mut pushed = 0;
            let push_until = |stream: &mut IvectorStream, upto: usize, pushed: &mut usize| {
                while *pushed < upto {
                    stream.push_frame(feats.row(*pushed));
                    *pushed += 1;
                }
            };
            forward_chunked(model, &feats, chunk_in, |_begin, end| {
                // the first accept step after which this chunk's output frames are ready
                let need = end + rctx;
                let mut ready_frames = None;
                for &f in &steps {
                    if f >= need {
                        ready_frames = Some(f);
                        break;
                    }
                }
                let (avail, finished) = match ready_frames {
                    Some(f) => (f, false),
                    None => (total, true),
                };
                push_until(&mut stream, avail, &mut pushed);
                if finished {
                    stream.set_input_finished();
                }
                let ready = stream.num_frames_ready();
                let mut iv = vec![0.0f32; ivdim];
                if ready > 0 {
                    let frame = (avail - 1).min(ready - 1);
                    stream.get_frame(frame, &mut iv);
                }
                iv
            })
        }
        m => panic!("unknown mode {m}"),
    };
    let t_net = t1.elapsed();
    let t2 = Instant::now();
    let dec = BatchDecoder {
        beam: model.conf.beam,
        acoustic_scale: model.conf.acoustic_scale,
        max_active: model.conf.max_active,
    };
    let (ids, cost) = dec.decode(graph, &model.tm.tid2pdf, &loglikes);
    let t_dec = t2.elapsed();
    let words: Vec<&str> = ids
        .iter()
        .map(|&i| model.word(i))
        .filter(|w| !w.is_empty() && *w != "<eps>")
        .collect();
    let mut s = String::new();
    s.push_str("{\"take\": ");
    utter::json::write_string(&mut s, wav.file_stem().unwrap().to_str().unwrap());
    s.push_str(&format!(", \"mode\": \"{mode}\", \"seconds\": {:.3}, \"frames\": {}, \"cost\": {:.3}, \"feat_ms\": {}, \"net_ms\": {}, \"decode_ms\": {}, \"words\": [",
        samples.len() as f64 / w.sample_rate as f64, feats.r, cost, t_feat.as_millis(), t_net.as_millis(), t_dec.as_millis()));
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        utter::json::write_string(&mut s, w);
    }
    let _ = sf;
    s.push_str("]}");
    s
}

fn main() {
    let args = parse_args();
    let t0 = Instant::now();
    let model = Model::open(&args.model).expect("model");
    eprintln!(
        "model loaded in {:?}: {} tids, {} pdfs, net in {} out {} ivector {} context {:?}, HCLr {} states {} arcs, {} words, frames_per_chunk {}",
        t0.elapsed(),
        model.tm.num_tids(),
        model.tm.num_pdfs,
        model.net.input_dim,
        model.net.output_dim,
        model.net.ivector_dim,
        model.net.context,
        model.hcl.num_states(),
        model.hcl.num_arcs(),
        model.words.len(),
        model.conf.frames_per_chunk
    );
    let grammar_text = std::fs::read_to_string(&args.grammar).expect("grammar file");
    let grammar = utter::json::parse_string_array(&grammar_text).expect("grammar JSON");
    let t1 = Instant::now();
    let graph = model
        .compile_grammar(&grammar, None, 5_000_000, |w| eprintln!("warning: {w}"))
        .expect("compose");
    eprintln!(
        "graph compiled in {:?}: {} states {} arcs",
        t1.elapsed(),
        graph.num_states(),
        graph.num_arcs()
    );

    let jobs: Vec<(PathBuf, String)> = args
        .wavs
        .iter()
        .flat_map(|w| args.modes.iter().map(move |m| (w.clone(), m.clone())))
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = std::sync::Mutex::new(Vec::<(usize, String)>::new());
    std::thread::scope(|sc| {
        for _ in 0..args.threads.min(jobs.len().max(1)) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= jobs.len() {
                    break;
                }
                let (wav, mode) = &jobs[i];
                let line = decode_take(&model, &graph, wav, mode, args.dither, args.step_ms);
                eprintln!("{}", line);
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
