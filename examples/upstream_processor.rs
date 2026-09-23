//! A host that runs an upstream waveform processor, such as a speaker extractor, in front of a
//! recognizer and lends it the processor's normalized samples, hop by hop or gathered into fixed
//! 40 ms blocks.
//!
//!     cargo run --release --example upstream_processor -- MODEL_DIR WAV [--hops]
//!
//! The processor is a stand-in that returns its input in 10 ms hops behind 30 ms of lookahead.
//! The WAV, 16 kHz mono, is captured 50 ms at a time with a second of silence after it, and the
//! capture loses half a second in the middle.
// [[rr:TD-13#Sample count defines time and delivery cadence]]
// [[rr:TD-13#The host composes two independent stream lifecycles]]

use std::path::Path;
use utter::{Model, Recognizer};

const HOP: usize = 160;
const BLOCK: usize = 640;

/// The stand-in: every sample it is given comes back once, normalized and in order.
#[derive(Default)]
struct Processor {
    held: Vec<f32>,
}

impl Processor {
    const LOOKAHEAD: usize = 480;

    fn push(&mut self, capture: &[i16], mut out: impl FnMut(&[f32])) {
        self.held
            .extend(capture.iter().map(|&s| f32::from(s) / 32768.0));
        while self.held.len() >= Self::LOOKAHEAD + HOP {
            out(&self.held[..HOP]);
            self.held.drain(..HOP);
        }
    }

    /// The end of input: what the lookahead held comes out, the last hop short.
    fn finish(&mut self, mut out: impl FnMut(&[f32])) {
        for hop in self.held.chunks(HOP) {
            out(hop);
        }
        self.held.clear();
    }
}

/// A recognizer over one stretch of unbroken capture, and the 40 ms block it is fed from.
struct Segment<'m> {
    rec: Recognizer<'m>,
    /// The source sample the recognizer's sample 0 was captured at.
    offset: u64,
    hops: bool,
    block: [f32; BLOCK],
    filled: usize,
    partial: String,
}

impl Segment<'_> {
    fn deliver(&mut self, mut hop: &[f32]) {
        if self.hops {
            return self.send(hop);
        }
        while !hop.is_empty() {
            let n = hop.len().min(BLOCK - self.filled);
            self.block[self.filled..self.filled + n].copy_from_slice(&hop[..n]);
            self.filled += n;
            hop = &hop[n..];
            if self.filled == BLOCK {
                let block = self.block;
                self.send(&block);
                self.filled = 0;
            }
        }
    }

    fn send(&mut self, samples: &[f32]) {
        let step = self
            .rec
            .accept_f32(samples)
            .expect("the processor keeps its output within full scale");
        let at = self.offset + step.sample;
        if step.endpoint {
            println!("final at source sample {at}: {}", self.rec.result());
        } else {
            let partial = self.rec.partial();
            if partial != self.partial {
                println!("partial at source sample {at}: {partial}");
                self.partial = partial.to_string();
            }
        }
    }

    /// The processor drains first, then the part-filled block goes, and only then does the
    /// recognizer flush.
    fn close(mut self, mut processor: Processor) {
        processor.finish(|hop| self.deliver(hop));
        if self.filled > 0 {
            let block = self.block;
            self.send(&block[..self.filled]);
        }
        println!("flush: {}", self.rec.final_result());
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(model_dir), Some(wav)) = (args.first(), args.get(1)) else {
        eprintln!("usage: upstream_processor MODEL_DIR WAV [--hops]");
        std::process::exit(2);
    };
    let model = Model::open(Path::new(model_dir))?;
    let grammar: Vec<String> = [
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "one", "two",
        "three", "four", "five", "six", "seven", "eight", "nine", "zero",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let wav = utter::wav::read_wav(Path::new(wav))?;
    assert_eq!(wav.sample_rate, 16000, "a 16 kHz WAV");
    let mut capture = wav.samples;
    // A second of digital silence before the end of input.
    capture.extend(vec![0; 16000]);
    let lost = capture.len() / 2..capture.len() / 2 + 8000;
    for (offset, stretch) in [
        (0, &capture[..lost.start]),
        (lost.end, &capture[lost.end..]),
    ] {
        // Each side of the lost half second is a segment of its own.
        let mut segment = Segment {
            rec: Recognizer::new(&model, 16_000.0, &grammar)?,
            offset: offset as u64,
            hops: args.iter().any(|a| a == "--hops"),
            block: [0.0; BLOCK],
            filled: 0,
            partial: String::new(),
        };
        segment.rec.set_words(true);
        // Samples in results count from here.
        println!("segment from source sample {offset}");
        let mut processor = Processor::default();
        for read in stretch.chunks(800) {
            processor.push(read, |hop| segment.deliver(hop));
        }
        segment.close(processor);
    }
    Ok(())
}
