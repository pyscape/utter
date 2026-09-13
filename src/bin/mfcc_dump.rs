//! Gate G1: stream a WAV through the online MFCC front end and write the frames as raw
//! little-endian f32, the shape `[[rr:read_matrix_dump]]` reads: a 2 x i32 header of rows and
//! columns, then the frames.
// [[rr:TD-2#Verification and acceptance]]

use std::io::Write;
use utter::frontend::OnlineMfcc;
use utter::mfcc::MfccOptions;
use utter::wav::read_wav;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let conf = std::fs::read_to_string(&args[0]).expect("mfcc.conf");
    let mut opts = MfccOptions::from_conf(&conf);
    opts.dither = 0.0;
    let wav = read_wav(std::path::Path::new(&args[1])).expect("wav");
    let block: usize = args.get(3).and_then(|b| b.parse().ok()).unwrap_or(640);
    let mut mfcc = OnlineMfcc::new(&opts);
    let samples: Vec<f32> = wav.samples.iter().map(|&s| s as f32).collect();
    for chunk in samples.chunks(block) {
        mfcc.accept(chunk);
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[2]).expect("out"));
    let rows = i32::try_from(mfcc.frames.len()).expect("header dimension fits i32");
    let cols = i32::try_from(mfcc.dim()).expect("header dimension fits i32");
    out.write_all(&rows.to_le_bytes()).unwrap();
    out.write_all(&cols.to_le_bytes()).unwrap();
    for f in &mfcc.frames {
        for v in f {
            out.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}
