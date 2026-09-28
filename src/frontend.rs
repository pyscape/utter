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
    /// Frames from `frames_first` on; those before it are gone.
    frames: Vec<Vec<f32>>,
    frames_first: usize,
    /// Each frame's log mel energies from `logmel_first` on, until the recognizer takes them;
    /// kept only in the recognizer's pipeline.
    keep_logmel: bool,
    logmel: Vec<f32>,
    logmel_first: usize,
    input_finished: bool,
}

impl OnlineMfcc {
    pub fn new(opts: &MfccOptions) -> Self {
        OnlineMfcc {
            mfcc: Mfcc::new(opts),
            remainder: Vec::new(),
            remainder_offset: 0,
            frames: Vec::new(),
            frames_first: 0,
            keep_logmel: false,
            logmel: Vec::new(),
            logmel_first: 0,
            input_finished: false,
        }
    }

    /// A stream whose first frame is `frame`: its samples begin at that frame's start.
    pub fn starting_at(opts: &MfccOptions, frame: usize) -> Self {
        let mut m = OnlineMfcc::new(opts);
        m.frames_first = frame;
        m.logmel_first = frame;
        m.remainder_offset = frame * m.mfcc.frame_shift;
        m
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
            let start = self.num_frames_ready() * shift;
            if start < self.remainder_offset {
                unreachable!("frame start before the retained samples");
            }
            let local = start - self.remainder_offset;
            if local + len > self.remainder.len() {
                break;
            }
            self.mfcc
                .compute_frame(&self.remainder[local..local + len], &mut row);
            self.frames.push(row.clone());
            if self.keep_logmel {
                self.logmel.extend_from_slice(self.mfcc.last_logmel());
            }
        }
        // Drop samples no future frame will read.
        let next_start = self.num_frames_ready() * shift;
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
        self.frames_first + self.frames.len()
    }

    pub fn frame(&self, t: usize) -> &[f32] {
        &self.frames[t - self.frames_first]
    }

    /// The frames kept, and the frame the first of them is.
    pub fn frames(&self) -> (&[Vec<f32>], usize) {
        (&self.frames, self.frames_first)
    }

    /// Mel bands per frame.
    pub fn num_bins(&self) -> usize {
        self.mfcc.num_bins()
    }

    /// Hand each frame's log mel energies from `from` up to `to` to `f` in order, and forget
    /// those before `to`. Frames before `from` are forgotten unread.
    pub fn take_logmel(&mut self, from: usize, to: usize, mut f: impl FnMut(&[f32])) {
        let bins = self.mfcc.num_bins();
        let end = to.min(self.logmel_first + self.logmel.len() / bins);
        let from = from.max(self.logmel_first);
        for t in from..end {
            let at = (t - self.logmel_first) * bins;
            f(&self.logmel[at..at + bins]);
        }
        let n = end.saturating_sub(self.logmel_first);
        self.logmel.drain(..n * bins);
        self.logmel_first += n;
    }

    /// Frames before `frame` will not be read again.
    pub fn forget_before(&mut self, frame: usize) {
        let n = frame
            .saturating_sub(self.frames_first)
            .min(self.frames.len());
        self.frames.drain(..n);
        self.frames_first += n;
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
        let mut mfcc = OnlineMfcc::new(opts);
        mfcc.keep_logmel = true;
        FeaturePipeline {
            mfcc,
            ivector: ivector.map(IvectorStream::new),
            ivector_pushed: 0,
        }
    }

    pub fn accept(&mut self, samples: &[f32]) {
        self.mfcc.accept(samples);
        if let Some(iv) = self.ivector.as_mut() {
            while self.ivector_pushed < self.mfcc.num_frames_ready() {
                iv.push_frame(self.mfcc.frame(self.ivector_pushed));
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
