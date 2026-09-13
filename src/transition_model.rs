// Vendored from Vosk-Rust (Apache-2.0), see third_party/Vosk-Rust/NOTICE.
//! Kaldi TransitionModel for a **chain** model: transition id to pdf and to phone.
//!
//! Chain models store `<Tuples>` of `(phone, hmm_state, forward_pdf, self_loop_pdf)`. With the
//! standard chain topology (one emitting state, two transitions), each tuple is one
//! transition-state with exactly 2 transition-ids: `2*i+1` = self-loop (self_loop_pdf) and
//! `2*i+2` = forward (forward_pdf), numbered from 1. The tuples are reached by seeking the
//! `<Tuples>` token; the topology before it is pure binary with no tokens.

use crate::kaldi_io::{err, find, KaldiReader};
use std::io::Result;

pub struct TransitionModel {
    /// tid2pdf\[transition_id\] = pdf_id; index 0 unused.
    pub tid2pdf: Vec<i32>,
    /// tid2phone\[transition_id\] = phone; index 0 unused.
    pub tid2phone: Vec<i32>,
    pub num_pdfs: usize,
    pub num_transition_states: usize,
}

impl TransitionModel {
    pub fn parse(data: &[u8]) -> Result<TransitionModel> {
        let marker = b"<Tuples> ";
        let pos = find(data, marker, 0).ok_or_else(|| err("no <Tuples>"))? + marker.len();
        let tail = &data[pos..];
        let mut kr = KaldiReader::new(tail);
        let n = kr.read_dim()?;
        // Each tuple is four size-byte-prefixed i32s; the tables are twice as long as the
        // count, so a count the file cannot back would size them from the file alone.
        if n.checked_mul(20).is_none_or(|need| need > tail.len()) {
            return Err(err("more tuples than the transition model holds"));
        }
        let mut tid2pdf = vec![-1i32; 2 * n + 1];
        let mut tid2phone = vec![0i32; 2 * n + 1];
        let mut num_pdfs = 0i32;
        for i in 0..n {
            let phone = kr.read_i32()?;
            let _hmm_state = kr.read_i32()?;
            let fwd = kr.read_i32()?;
            let slf = kr.read_i32()?;
            if fwd < 0 || slf < 0 {
                return Err(err("negative pdf id in a transition tuple"));
            }
            num_pdfs = num_pdfs
                .max(fwd.saturating_add(1))
                .max(slf.saturating_add(1));
            tid2pdf[2 * i + 1] = slf;
            tid2pdf[2 * i + 2] = fwd;
            tid2phone[2 * i + 1] = phone;
            tid2phone[2 * i + 2] = phone;
        }
        Ok(TransitionModel {
            tid2pdf,
            tid2phone,
            num_pdfs: usize::try_from(num_pdfs).unwrap_or(0),
            num_transition_states: n,
        })
    }

    pub fn num_tids(&self) -> usize {
        self.tid2pdf.len() - 1
    }
}
