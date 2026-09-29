//! Chunked network execution after Kaldi's `DecodableNnetLoopedOnline`: output frames become
//! ready a chunk at a time once the chunk's input frames and the right context exist, the
//! first frame is repeated for the left context, the last frame once input has finished, and
//! each chunk takes the newest i-vector available when it is computed. Layers keep their rows
//! between chunks, so a chunk costs its own rows only.
// [[rr:TD-2#The network]]

use crate::frontend::FeaturePipeline;
use crate::model::Model;
use crate::nnet3::{InputView, Streamer};

pub struct LoopedNnet<'m> {
    model: &'m Model,
    streamer: Streamer<'m>,
    /// Log-likelihoods, acoustic scale applied, one row per output frame from `outputs_first`:
    /// the rows the decoder has not yet read.
    outputs: Vec<f32>,
    outputs_first: usize,
    /// The frame `frame` last served; rows before it, and before `hold`, retire at the next
    /// chunk.
    read: usize,
    hold: Option<usize>,
    chunks_computed: usize,
    /// Output frames before this one belong to earlier utterances on the same pipeline.
    frame_offset: usize,
}

impl<'m> LoopedNnet<'m> {
    pub fn new(model: &'m Model) -> Self {
        LoopedNnet {
            model,
            streamer: model.net.streamer(),
            outputs: Vec::new(),
            outputs_first: 0,
            read: 0,
            hold: None,
            chunks_computed: 0,
            frame_offset: 0,
        }
    }

    pub fn set_frame_offset(&mut self, offset: usize) {
        self.frame_offset = offset;
    }

    pub fn frame_offset(&self) -> usize {
        self.frame_offset
    }

    pub fn output_dim(&self) -> usize {
        self.model.net.output_dim
    }

    /// Output frames ready for the current utterance, Kaldi's `NumFramesReady`.
    pub fn num_frames_ready(&self, pipe: &FeaturePipeline) -> usize {
        let features_ready = pipe.mfcc.num_frames_ready();
        if features_ready == 0 {
            return 0;
        }
        let sf = self.model.conf.frame_subsampling_factor;
        let total = if pipe.mfcc.is_input_finished() {
            features_ready.div_ceil(sf)
        } else {
            let rctx = self.model.net.context.1;
            let non_subsampled = features_ready.saturating_sub(rctx);
            let chunk = self.model.conf.frames_per_chunk;
            (non_subsampled / chunk) * chunk / sf
        };
        total.saturating_sub(self.frame_offset)
    }

    /// Keep the rows from absolute frame `frame` on past the next chunk, for [`row`](Self::row).
    pub fn hold_from(&mut self, frame: usize) {
        self.hold = Some(frame);
    }

    /// The row of absolute frame `frame`, if it is computed and has not retired.
    pub fn row(&self, frame: usize) -> Option<&[f32]> {
        if frame < self.outputs_first || frame >= self.outputs_end() {
            return None;
        }
        let dim = self.output_dim();
        let at = (frame - self.outputs_first) * dim;
        Some(&self.outputs[at..at + dim])
    }

    fn outputs_end(&self) -> usize {
        self.outputs_first + self.outputs.len() / self.output_dim()
    }

    /// Log-likelihood row for utterance frame `frame`, computing chunks as needed. Frames are
    /// asked for in order: rows before the last one asked for are gone.
    pub fn frame(&mut self, pipe: &mut FeaturePipeline, frame: usize) -> &[f32] {
        let absolute = frame + self.frame_offset;
        assert!(
            absolute >= self.read,
            "output frames asked for out of order"
        );
        self.read = absolute;
        while self.outputs_end() <= absolute {
            self.advance_chunk(pipe);
        }
        let dim = self.output_dim();
        let at = (absolute - self.outputs_first) * dim;
        &self.outputs[at..at + dim]
    }

    fn advance_chunk(&mut self, pipe: &mut FeaturePipeline) {
        let net = &self.model.net;
        let (lctx, rctx) = net.context;
        let chunk = self.model.conf.frames_per_chunk;
        let sf = self.model.conf.frame_subsampling_factor;
        let begin = self.chunks_computed * chunk;
        let end = begin + chunk;
        let ready = pipe.mfcc.num_frames_ready();
        assert!(ready > 0, "no features to run the network on");
        let finished = pipe.mfcc.is_input_finished();
        if end + rctx > ready && !finished {
            panic!("network chunk requested before its input frames exist");
        }
        let mut ivector = vec![0.0f32; net.ivector_dim];
        if net.ivector_dim > 0 {
            if let Some(iv) = pipe.ivector.as_mut() {
                let iv_ready = iv.num_frames_ready();
                if iv_ready > 0 {
                    let frame = (ready - 1).min(iv_ready - 1);
                    iv.get_frame(frame, &mut ivector);
                }
            }
        }
        // Kaldi's chunk sees input rows up to the chunk end plus the right context, no further,
        // even when more frames are ready.
        let (frames, base) = pipe.mfcc.frames();
        let view = InputView {
            frames,
            base,
            start: 0,
            ready,
            finished,
            limit: (end + rctx) as i64,
            first: -(lctx as i64),
        };
        let dim = self.output_dim();
        let retire = self
            .read
            .min(self.hold.unwrap_or(usize::MAX))
            .min(self.outputs_end())
            .saturating_sub(self.outputs_first);
        self.outputs.drain(..retire * dim);
        self.outputs_first += retire;
        let scale = self.model.conf.acoustic_scale;
        let (first, rows, cols) = self.streamer.advance(&view, &ivector);
        assert_eq!(cols, dim, "output rows are not the network's output width");
        for (i, row) in rows.chunks_exact(cols).enumerate() {
            let Ok(t) = usize::try_from(first + i as i64) else {
                continue;
            };
            if t.is_multiple_of(sf) {
                assert_eq!(
                    t / sf,
                    self.outputs_first + self.outputs.len() / dim,
                    "output rows out of order"
                );
                self.outputs.extend(row.iter().map(|v| v * scale));
            }
        }
        self.chunks_computed += 1;
        pipe.mfcc
            .forget_before((self.chunks_computed * chunk).saturating_sub(lctx));
        assert!(
            self.outputs_end() >= end / sf,
            "chunk produced too few output rows"
        );
    }
}
