//! Kaldi's `OnlineSilenceWeighting`: frames the decoder's current best path labels as silence
//! phones enter the i-vector statistics at a small weight, retroactively over the last hundred
//! decoder frames when the traceback changes, and frames newer than the traceback take the most
//! recent weight, or the silence weight when there is none yet.
// [[rr:TD-2#Front end: the i-vector branch]]

use crate::decoder::Path;

#[derive(Clone, Copy)]
struct FrameInfo {
    known: bool,
    is_silence: bool,
    current_weight: f32,
}

pub struct SilenceWeighting {
    silence_phones: Vec<i32>,
    silence_weight: f32,
    subsampling: usize,
    frame_info: Vec<FrameInfo>,
}

impl SilenceWeighting {
    pub fn new(silence_phones: &[i32], silence_weight: f32, subsampling: usize) -> Self {
        SilenceWeighting { silence_phones: silence_phones.to_vec(), silence_weight, subsampling, frame_info: Vec::new() }
    }

    pub fn active(&self) -> bool {
        !self.silence_phones.is_empty() && self.silence_weight != 1.0
    }

    /// Record the silence status of every decoded frame from the best path's phone segments.
    pub fn compute_current_traceback(&mut self, path: &Path, num_frames_decoded: usize) {
        if self.frame_info.len() < num_frames_decoded {
            self.frame_info.resize(num_frames_decoded, FrameInfo { known: false, is_silence: false, current_weight: 0.0 });
        }
        for seg in &path.phones {
            let sil = self.silence_phones.contains(&seg.phone);
            for f in seg.start..seg.end.min(num_frames_decoded) {
                self.frame_info[f].known = true;
                self.frame_info[f].is_silence = sil;
            }
        }
    }

    /// Weight changes per input frame, Kaldi's `GetDeltaWeights`.
    pub fn get_delta_weights(&mut self, num_frames_ready: usize, first_decoder_frame: usize) -> Vec<(usize, f32)> {
        let fs = self.subsampling;
        let mut deltas = Vec::new();
        if num_frames_ready <= first_decoder_frame {
            return deltas;
        }
        let num_decoder_frames_ready = (num_frames_ready - first_decoder_frame + fs - 1) / fs;
        let prev = self.frame_info.len();
        if prev < num_decoder_frames_ready {
            self.frame_info.resize(num_decoder_frames_ready, FrameInfo { known: false, is_silence: false, current_weight: 0.0 });
        }
        let begin = prev.saturating_sub(100);
        let frames_out = self.frame_info.len() - begin;
        if frames_out == 0 {
            return deltas;
        }
        let mut weight = vec![1.0f32; frames_out];
        if !self.frame_info[begin].known {
            let w = if begin == 0 { self.silence_weight } else { self.frame_info[begin - 1].current_weight };
            for v in weight.iter_mut() {
                *v = w;
            }
        } else {
            for offset in 0..frames_out {
                let fi = self.frame_info[begin + offset];
                if !fi.known {
                    weight[offset] = weight[offset - 1];
                } else if fi.is_silence {
                    weight[offset] = self.silence_weight;
                }
            }
        }
        for offset in 0..frames_out {
            let frame = begin + offset;
            let old = self.frame_info[frame].current_weight;
            let new = weight[offset];
            let diff = new - old;
            self.frame_info[frame].current_weight = new;
            if diff != 0.0 || offset + 1 == frames_out {
                for i in 0..fs {
                    deltas.push((first_decoder_frame + frame * fs + i, diff));
                }
            }
        }
        deltas
    }
}
