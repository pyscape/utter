//! The streaming feature pipeline after Kaldi's `OnlineNnet2FeaturePipeline`: MFCC frames
//! emitted as soon as their window is complete, and the i-vector branch over them.
// [[rr:TD-2#Front end: MFCC]]
// [[rr:TD-2#Front end: the i-vector branch]]

use crate::ivector::{IvectorInfo, IvectorStream};
use crate::mfcc::{Mfcc, MfccOptions};

pub struct OnlineMfcc {
    mfcc: Mfcc,
    /// Samples not yet consumed by a complete frame, plus the overlap later frames need.
    remainder: Vec<f32>,
    /// Sample index of `remainder[0]` in the stream.
    remainder_offset: usize,
    pub frames: Vec<Vec<f32>>,
    input_finished: bool,
}

impl OnlineMfcc {
    pub fn new(opts: &MfccOptions) -> Self {
        OnlineMfcc { mfcc: Mfcc::new(opts), remainder: Vec::new(), remainder_offset: 0, frames: Vec::new(), input_finished: false }
    }

    pub fn dim(&self) -> usize {
        self.mfcc.dim()
    }

    /// Feed samples in the int16 range; every frame whose window is complete is computed.
    pub fn accept(&mut self, samples: &[f32]) {
        self.remainder.extend_from_slice(samples);
        let (len, shift) = (self.mfcc.frame_len, self.mfcc.frame_shift);
        let mut row = vec![0.0f32; self.mfcc.dim()];
        loop {
            let start = self.frames.len() * shift;
            if start < self.remainder_offset {
                unreachable!("frame start before the retained samples");
            }
            let local = start - self.remainder_offset;
            if local + len > self.remainder.len() {
                break;
            }
            self.mfcc.compute_frame(&self.remainder[local..local + len], &mut row);
            self.frames.push(row.clone());
        }
        // Drop samples no future frame will read.
        let next_start = self.frames.len() * shift;
        if next_start > self.remainder_offset {
            let drop = next_start - self.remainder_offset;
            self.remainder.drain(..drop.min(self.remainder.len()));
            self.remainder_offset = next_start;
        }
    }

    pub fn input_finished(&mut self) {
        self.input_finished = true;
    }

    pub fn is_input_finished(&self) -> bool {
        self.input_finished
    }

    pub fn num_frames_ready(&self) -> usize {
        self.frames.len()
    }
}

/// MFCC plus the i-vector branch, sharing the frames.
pub struct FeaturePipeline<'a> {
    pub mfcc: OnlineMfcc,
    pub ivector: Option<IvectorStream<'a>>,
    ivector_pushed: usize,
}

impl<'a> FeaturePipeline<'a> {
    pub fn new(opts: &MfccOptions, ivector: Option<&'a IvectorInfo>) -> Self {
        FeaturePipeline { mfcc: OnlineMfcc::new(opts), ivector: ivector.map(IvectorStream::new), ivector_pushed: 0 }
    }

    pub fn accept(&mut self, samples: &[f32]) {
        self.mfcc.accept(samples);
        if let Some(iv) = self.ivector.as_mut() {
            while self.ivector_pushed < self.mfcc.frames.len() {
                iv.push_frame(&self.mfcc.frames[self.ivector_pushed]);
                self.ivector_pushed += 1;
            }
        }
    }

    pub fn input_finished(&mut self) {
        self.mfcc.input_finished();
        if let Some(iv) = self.ivector.as_mut() {
            iv.set_input_finished();
        }
    }
}
