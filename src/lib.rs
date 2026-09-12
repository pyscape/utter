//! utter: a pure-Rust streaming decoder for Vosk models.
//!
//! `[[rr:TD-2]]`

pub mod compose;
pub mod decode;
pub mod decoder;
pub mod fft;
pub mod frontend;
pub mod fst;
pub mod gemm;
pub mod grammar;
pub mod ivector;
pub mod json;
pub mod kaldi_io;
pub mod looped;
pub mod mfcc;
pub mod model;
pub mod nnet3;
pub mod recognizer;
pub mod silence_weighting;
pub mod transition_model;
pub mod wav;

pub use model::Model;
pub use recognizer::{Recognizer, Step};
