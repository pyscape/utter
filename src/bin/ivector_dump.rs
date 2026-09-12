//! Gate G1, i-vector half: stream a WAV through the online feature pipeline and write the
//! i-vector estimate every `--period` frames in the shape `[[rr:read_matrix_dump]]` reads and
//! `ivector-extract-online2 --ivector-period` writes, for `[[rr:ivector_half]]` to compare.
// [[rr:TD-2#Verification and acceptance]]
// [[rr:i-vector: against the wheel's Kaldi, passed once its quiet-frame rule was matched]]

use std::io::Write;
use utter::frontend::FeaturePipeline;
use utter::wav::read_wav;
use utter::Model;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: ivector_dump MODEL_DIR WAV OUT [--period N] [--block N]");
        std::process::exit(2);
    }
    let mut period = 10usize;
    let mut block = 640usize;
    let mut it = args[3..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--period" => period = it.next().unwrap().parse().unwrap(),
            "--block" => block = it.next().unwrap().parse().unwrap(),
            other => panic!("unknown argument {other}"),
        }
    }
    let model = Model::open(std::path::Path::new(&args[0])).expect("model");
    let info = model
        .ivector
        .as_ref()
        .expect("model has no i-vector extractor");
    let mut opts = model.mfcc_opts.clone();
    opts.dither = 0.0;
    let wav = read_wav(std::path::Path::new(&args[1])).expect("wav");
    let samples: Vec<f32> = wav.samples.iter().map(|&s| s as f32).collect();
    let mut pipe = FeaturePipeline::new(&opts, Some(info));
    for chunk in samples.chunks(block) {
        pipe.accept(chunk);
    }
    pipe.input_finished();
    let iv = pipe.ivector.as_mut().expect("i-vector branch");
    let ready = iv.num_frames_ready();
    let dim = info.ivector_dim();
    let mut row = vec![0.0f32; dim];
    let frames: Vec<usize> = (0..ready).step_by(period).collect();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[2]).expect("out"));
    out.write_all(&(frames.len() as i32).to_le_bytes()).unwrap();
    out.write_all(&(dim as i32).to_le_bytes()).unwrap();
    for f in frames {
        iv.get_frame(f, &mut row);
        for v in &row {
            out.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}
