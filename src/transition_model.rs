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
    /// tid2pdf[transition_id] = pdf_id; index 0 unused.
    pub tid2pdf: Vec<i32>,
    /// tid2phone[transition_id] = phone; index 0 unused.
    pub tid2phone: Vec<i32>,
    pub num_pdfs: usize,
    pub num_transition_states: usize,
}

impl TransitionModel {
    pub fn parse(data: &[u8]) -> Result<TransitionModel> {
        let marker = b"<Tuples> ";
        let pos = find(data, marker, 0).ok_or_else(|| err("no <Tuples>"))? + marker.len();
        let mut kr = KaldiReader::new(&data[pos..]);
        let n = kr.read_i32()? as usize;
        let mut tid2pdf = vec![-1i32; 2 * n + 1];
        let mut tid2phone = vec![0i32; 2 * n + 1];
        let mut num_pdfs = 0i32;
        for i in 0..n {
            let phone = kr.read_i32()?;
            let _hmm_state = kr.read_i32()?;
            let fwd = kr.read_i32()?;
            let slf = kr.read_i32()?;
            num_pdfs = num_pdfs.max(fwd + 1).max(slf + 1);
            tid2pdf[2 * i + 1] = slf;
            tid2pdf[2 * i + 2] = fwd;
            tid2phone[2 * i + 1] = phone;
            tid2phone[2 * i + 2] = phone;
        }
        Ok(TransitionModel {
            tid2pdf,
            tid2phone,
            num_pdfs: num_pdfs as usize,
            num_transition_states: n,
        })
    }

    pub fn num_tids(&self) -> usize {
        self.tid2pdf.len() - 1
    }
}
