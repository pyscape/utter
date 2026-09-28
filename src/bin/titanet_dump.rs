//! The TitaNet front end and network against kaldi-native-fbank, onnxruntime and sherpa-onnx,
//! for scripts/titanet_oracle.py.
//!
//!     titanet_dump MODEL net FEATURES OUT        embeddings of normalised feature matrices
//!     titanet_dump MODEL audio SPANS OUT         log mel features and embeddings of spans
//!
//! FEATURES: u32 count, then per matrix u32 frames, u32 bands (80) and the f32 values.
//! SPANS: u32 count, then per span u32 sample rate, u32 samples and the f32 samples.
//! `net` writes OUT, the embeddings end to end as f32. `audio` writes OUT.emb (embeddings),
//! and OUT.fbank, the log mel features before normalisation of every 16 kHz span, in the
//! FEATURES layout; spans at other rates are embedded through the resampler and have no
//! features there.

use std::fs;
use utter::fbank::{self, Fbank, NUM_BANDS};
use utter::titanet::{Embedding, TitaNet};

fn u32_at(b: &[u8], at: &mut usize) -> usize {
    let v = u32::from_le_bytes(b[*at..*at + 4].try_into().expect("u32"));
    *at += 4;
    v as usize
}

fn f32s_at(b: &[u8], at: &mut usize, n: usize) -> Vec<f32> {
    let v = b[*at..*at + 4 * n]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("f32")))
        .collect();
    *at += 4 * n;
    v
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u32_bytes(n: usize) -> [u8; 4] {
    u32::try_from(n).expect("fits a u32").to_le_bytes()
}

// A span's rate is tens of thousands of hertz, exact in an f32.
#[allow(clippy::cast_precision_loss)]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, mode, input, out] = &args[..] else {
        eprintln!("usage: titanet_dump MODEL net|audio INPUT OUT");
        std::process::exit(2);
    };
    let model = TitaNet::open(std::path::Path::new(dir)).expect("TitaNet model");
    let b = fs::read(input).expect("input");
    let mut at = 0;
    let n = u32_at(&b, &mut at);
    let mut emb = Vec::new();
    let mut job = Embedding::new();
    match mode.as_str() {
        "net" => {
            for _ in 0..n {
                let t = u32_at(&b, &mut at);
                let d = u32_at(&b, &mut at);
                assert_eq!(d, NUM_BANDS);
                let x = f32s_at(&b, &mut at, t * d);
                job.begin_features(&model, &x).expect("features");
                job.advance(&model, u64::MAX);
                emb.extend(job.take());
            }
            fs::write(out, bytes(&emb)).expect("write");
        }
        "audio" => {
            let mut fb = Fbank::new();
            let mut feats = Vec::new();
            let mut count = 0;
            for _ in 0..n {
                let rate = u32_at(&b, &mut at);
                let len = u32_at(&b, &mut at);
                let x = f32s_at(&b, &mut at, len);
                emb.extend(model.embed_at_rate(&x, rate as f32).expect("span"));
                if rate == fbank::SAMPLE_RATE as usize {
                    let t = fbank::num_frames(len);
                    let mut rows = vec![0.0; t * NUM_BANDS];
                    for (f, row) in rows.chunks_exact_mut(NUM_BANDS).enumerate() {
                        fb.frame(&x, f, row);
                    }
                    feats.extend(u32_bytes(t));
                    feats.extend(u32_bytes(NUM_BANDS));
                    feats.extend(bytes(&rows));
                    count += 1;
                }
            }
            fs::write(format!("{out}.emb"), bytes(&emb)).expect("write");
            let mut f = u32_bytes(count).to_vec();
            f.extend(feats);
            fs::write(format!("{out}.fbank"), f).expect("write");
        }
        _ => {
            eprintln!("mode is net or audio");
            std::process::exit(2);
        }
    }
}
