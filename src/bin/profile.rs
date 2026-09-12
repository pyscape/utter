//! Where a block's compute goes: network chunk, decoder frame, i-vector update, on one take.
use std::time::Instant;
use utter::frontend::FeaturePipeline;
use utter::looped::LoopedNnet;
use utter::wav::read_wav;
use utter::Model;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = Model::open(std::path::Path::new(&args[0])).unwrap();
    let grammar =
        utter::json::parse_string_array(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let wav = read_wav(std::path::Path::new(&args[2])).unwrap();
    let samples: Vec<f32> = wav.samples.iter().map(|&s| s as f32).collect();
    let mut opts = model.mfcc_opts.clone();
    opts.dither = 0.0;
    // network alone, chunk by chunk, with a zero i-vector and with the real one
    for with_iv in [false, true] {
        let mut pipe = FeaturePipeline::new(
            &opts,
            if with_iv {
                model.ivector.as_ref()
            } else {
                None
            },
        );
        let mut nnet = LoopedNnet::new(&model);
        let mut t_feat = 0f64;
        let mut t_net = Vec::new();
        let mut decoded = 0usize;
        for block in samples.chunks(640) {
            let t = Instant::now();
            pipe.accept(block);
            t_feat += t.elapsed().as_secs_f64();
            let ready = nnet.num_frames_ready(&pipe);
            if ready > decoded {
                let t = Instant::now();
                while decoded < ready {
                    let _ = nnet.frame(&mut pipe, decoded);
                    decoded += 1;
                }
                t_net.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        t_net.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = t_net.len();
        println!(
            "ivector={with_iv}: mfcc total {:.1} ms; network+ivector per chunk p50 {:.2} p95 {:.2} max {:.2} ms over {n} chunks",
            t_feat * 1000.0,
            t_net[n / 2],
            t_net[n * 95 / 100],
            t_net[n - 1]
        );
    }
    // decoder alone over the whole take
    let graph = std::sync::Arc::new(
        model
            .compile_grammar(&grammar, None, 5_000_000, |_| {})
            .unwrap(),
    );
    let mut pipe = FeaturePipeline::new(&opts, model.ivector.as_ref());
    let mut nnet = LoopedNnet::new(&model);
    pipe.accept(&samples);
    pipe.input_finished();
    let ready = nnet.num_frames_ready(&pipe);
    let rows: Vec<Vec<f32>> = (0..ready)
        .map(|f| nnet.frame(&mut pipe, f).to_vec())
        .collect();
    let cfg = utter::decoder::DecoderConfig {
        beam: model.conf.beam,
        max_active: model.conf.max_active,
        min_active: model.conf.min_active,
        beam_delta: 0.5,
    };
    let mut dec = utter::decoder::Decoder::new(graph, &model.tm.tid2pdf, &model.tm.tid2phone, cfg);
    let t = Instant::now();
    let mut max_active = 0;
    for r in &rows {
        dec.advance_frame(r);
        max_active = max_active.max(dec.num_active());
    }
    let el = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "decoder: {:.3} ms per frame ({} frames, max active {}), traceback {:.3} ms",
        el / rows.len() as f64,
        rows.len(),
        max_active,
        {
            let t = Instant::now();
            for _ in 0..100 {
                let _ = dec.best_path(false);
            }
            t.elapsed().as_secs_f64() * 10.0
        }
    );
}
