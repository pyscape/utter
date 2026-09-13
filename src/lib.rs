//! utter: a pure-Rust streaming decoder for Vosk models, built for applications that act on
//! partial results. Feed a [`Recognizer`] 16 kHz audio in small blocks and after every block it
//! tells you the words it believes so far, the rival readings it is still weighing, how loud and
//! how stable each word is, and when the speaker stopped.
// [[rr:TD-2]]
//!
//! # Quick start
//!
//! ```no_run
//! use std::path::Path;
//!
//! # fn blocks() -> Vec<Vec<i16>> { Vec::new() }
//! # fn main() -> std::io::Result<()> {
//! let model = utter::Model::open(Path::new("vosk-model-small-en-us-0.15"))?;
//! let grammar: Vec<String> = ["alpha", "bravo", "seven"]
//!     .iter()
//!     .map(|s| s.to_string())
//!     .collect();
//! let mut rec = utter::Recognizer::new(&model, 16_000.0, &grammar)?;
//! rec.set_words(true);         // word spans on finals
//! rec.set_partial_words(true); // and on partials
//! rec.set_alternatives(4);     // rivals to carry alongside the partial
//!
//! for block in blocks() {                // 40 ms of 16-bit mono PCM
//!     let step = rec.accept(&block);     // decodes, applies the endpoint rules
//!     if step.endpoint {
//!         println!("{}", rec.result());  // the segment that just ended
//!     } else {
//!         println!("{}", rec.partial()); // best path, rivals, word spans
//!     }
//! }
//! println!("{}", rec.final_result());
//! # Ok(())
//! # }
//! ```
//!
//! # Where the documentation is
//!
//! Every result is a JSON string, the same one the Python and C bindings return. The keys are
//! listed in the [results reference]; the same page is linked from each method that returns
//! one. If you are coming from the `vosk` package, [Coming from Vosk] maps its methods to this
//! crate's. What to do with a partial is in the [guide].
//!
//! The public API is [`Model`], [`Recognizer`] and the types in [`recognizer`]. The other
//! modules are the runtime behind them, public so the `stream` tool and the tests can reach
//! them, and hidden from these docs.
//!
//! [results reference]: https://github.com/pyscape/utter/blob/main/docs/reference/results.md
//! [Coming from Vosk]: https://github.com/pyscape/utter/blob/main/docs/reference/vosk.md
//! [guide]: https://github.com/pyscape/utter/blob/main/docs/guide/

// Index loops over parallel slices read as the mathematics does; the lint's iterator forms do not.
#![warn(missing_docs)]
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

pub mod capi;
#[doc(hidden)]
pub mod compose;
#[doc(hidden)]
pub mod decode;
#[doc(hidden)]
pub mod decoder;
#[doc(hidden)]
pub mod fft;
#[doc(hidden)]
pub mod frontend;
#[doc(hidden)]
pub mod fst;
#[doc(hidden)]
pub mod gemm;
#[doc(hidden)]
pub mod grammar;
#[doc(hidden)]
pub mod ivector;
#[doc(hidden)]
pub mod json;
#[doc(hidden)]
pub mod kaldi_io;
#[doc(hidden)]
pub mod looped;
#[doc(hidden)]
pub mod mfcc;
pub mod model;
#[doc(hidden)]
pub mod nnet3;
pub mod recognizer;
#[doc(hidden)]
pub mod silence_weighting;
#[doc(hidden)]
pub mod transition_model;
#[doc(hidden)]
pub mod wav;

/// The git revision of the checkout this crate was built from, with a `-dirty` suffix when
/// tracked files had uncommitted edits, or `unknown` when there was no checkout to ask.
// [[rr:build.rs#main]]
pub const REVISION: &str = env!("UTTER_REVISION");

pub use model::Model;
pub use recognizer::{Endpoint, Recognizer, RecognizerOptions, Step};
