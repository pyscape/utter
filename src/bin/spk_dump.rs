//! The speaker front end and network against Kaldi: stream a WAV through a speaker model and
//! write its features before and after mean normalisation as Kaldi text archives, then print the
//! vector pooled over the frames `nnet3-xvector-compute` pools without padding, beside the vector
//! over every frame.
// [[rr:TD-14#Verification]]

use std::io::Write;
use utter::speaker::{SpeakerModel, SpeakerStream};
use utter::wav::read_wav;

fn write_ark(path: &str, key: &str, frames: &[Vec<f32>]) {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path).expect("ark"));
    writeln!(out, "{key}  [").unwrap();
    for (i, f) in frames.iter().enumerate() {
        let row: Vec<String> = f.iter().map(|v| format!("{v:e}")).collect();
        let close = if i + 1 == frames.len() { " ]" } else { "" };
        writeln!(out, "  {}{close}", row.join(" ")).unwrap();
    }
}

// A WAV's rate is a few tens of thousands, exact in an f32.
#[allow(clippy::cast_precision_loss)]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = SpeakerModel::open(std::path::Path::new(&args[0])).expect("speaker model");
    let wav = read_wav(std::path::Path::new(&args[1])).expect("wav");
    let prefix = &args[2];
    let mut stream = SpeakerStream::new(&model, wav.sample_rate as f32).expect("stream");
    for block in wav.samples.chunks(640) {
        stream.accept(block);
    }
    stream.finish();
    let (raw, normalized) = stream.features();
    write_ark(&format!("{prefix}.mfcc.ark"), "utt", raw);
    write_ark(&format!("{prefix}.feats.ark"), "utt", normalized);
    let n = normalized.len();
    let json = |ev: Option<utter::speaker::Evidence>| match ev {
        Some(e) => format!(
            "{{\"frames\": {}, \"vector\": [{}]}}",
            e.frames,
            e.vector
                .iter()
                .map(|v| format!("{v:e}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        None => "null".into(),
    };
    println!(
        "{{\"num_frames\": {n}, \"unpadded\": {}, \"all\": {}}}",
        json(stream.pool(&[(7, n.saturating_sub(7))], |_| true)),
        json(stream.pool(&[(0, n)], |_| true))
    );
}
