//! The recognizer: libvosk's lifecycle over the streaming pipeline, the decoder, and the
//! result shapes libvosk emits, extended with sample-indexed intervals, per-word energy, hold
//! time, and partial alternatives.
// [[rr:TD-2#Interface]]
// [[rr:TD-2#The decoder: word intervals]]
// [[rr:TD-2#The decoder: per-word energy and silence]]
// [[rr:TD-2#The decoder: partial alternatives]]
// [[rr:TD-2#The decoder: finals]]

use crate::decoder::{Decoder, DecoderConfig, Path, CENSUS_NATS};
use crate::frontend::FeaturePipeline;
use crate::fst::{Label, VectorFst};
use crate::json::write_string;
use crate::looped::LoopedNnet;
use crate::model::{Model, WordBoundary};
use crate::silence_weighting::SilenceWeighting;
use crate::speaker::{SpeakerModel, SpeakerStream};
use crate::titanet::{Embedding, TitaNet, RECOGNIZER_MAX_FRAMES};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Initialized,
    Running,
    Endpoint,
    Finalized,
}

/// What one `accept` call reports.
#[derive(Clone, Copy, Debug, Default)]
pub struct Step {
    /// An endpoint rule fired after this call's audio was decoded.
    pub endpoint: bool,
    /// Sample position, in audio fed since construction, of the last decoded frame's end.
    pub sample: u64,
}

/// A word of a path aligned to its frames.
#[derive(Clone, Debug)]
pub struct WordSpan {
    /// The word's id in the model's table.
    pub word: Label,
    /// First output frame of the word within the current utterance.
    pub start_frame: usize,
    /// One past the word's last output frame.
    pub end_frame: usize,
}

/// Default size of the composed graph a grammar may reach before construction refuses it.
pub const DEFAULT_MAX_GRAPH_STATES: usize = 5_000_000;

/// libvosk's weight for silence frames in the i-vector statistics.
pub const SILENCE_WEIGHT: f32 = 1e-3;

/// Largest grammar accepted: distinct words, counted before unknown words are dropped.
pub const MAX_GRAMMAR_WORDS: usize = 300;

/// The token a reading with no word carries.
pub const SIL: &str = "[sil]";
/// The reading of a best path that carries no word and ends inside a word's phones: a word has
/// begun that the grammar cannot yet tell apart from its neighbours.
pub const SPEECH: &str = "[speech]";

/// Construction options beyond libvosk's.
#[derive(Clone, Debug)]
pub struct RecognizerOptions {
    /// Add the model's unknown-word symbol to the grammar with this cost on its arcs.
    pub unknown_cost: Option<f32>,
    /// States the composed graph may reach before construction refuses the grammar.
    pub max_graph_states: usize,
    /// Weight for frames the decoder's best path calls silence; 1.0 turns it off.
    // [[rr:i-vector: against the wheel's Kaldi, passed once its quiet-frame rule was matched]]
    pub silence_weight: f32,
    /// One endpoint rule of the host's beside the model's, off by default; the switch for
    /// experiments that call a final earlier than Kaldi's rules, at a parity cost G5 measures.
    pub endpoint_rule: Option<crate::model::EndpointRule>,
}

impl Default for RecognizerOptions {
    fn default() -> Self {
        RecognizerOptions {
            unknown_cost: None,
            max_graph_states: DEFAULT_MAX_GRAPH_STATES,
            silence_weight: SILENCE_WEIGHT,
            endpoint_rule: None,
        }
    }
}

/// What closed an utterance: one of the model's numbered rules, the host's bound, the flush of
/// `final_result`, or the host calling `result` with no rule fired. [`label`](Self::label) is the
/// final's `endpoint` value: <https://github.com/pyscape/utter/blob/main/docs/reference/results.md#the-endpoint-value>.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    /// The model's rule of this number fired, `1` to `5`.
    Rule(u8),
    /// The host's bound from [`set_endpoint_bound`](Recognizer::set_endpoint_bound) fired.
    Bound,
    /// The host's bound fired on a wordless path at floor energy, with the margin of
    /// [`set_endpoint_floor_margin`](Recognizer::set_endpoint_floor_margin); the final is the
    /// path as it stood, with no word forced onto it.
    Floor,
    /// `final_result` flushed the pipeline.
    Flush,
    /// The host called `result` with no rule fired.
    Host,
}

impl Endpoint {
    /// The final's `endpoint` value: `rule1`..`rule5`, `bound`, `floor`, `flush` or `host`.
    pub fn label(self) -> String {
        match self {
            Endpoint::Rule(n) => format!("rule{n}"),
            Endpoint::Bound => "bound".into(),
            Endpoint::Floor => "floor".into(),
            Endpoint::Flush => "flush".into(),
            Endpoint::Host => "host".into(),
        }
    }
}

/// What an entry of a word list spans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryWord {
    /// A word of the grammar, by id.
    Word(Label),
    /// A run of silence phones.
    Silence,
    /// The phones of a word the path has entered and not yet named.
    Speech,
}

/// One entry of a word list.
#[derive(Clone, Debug)]
pub struct Entry {
    /// What the entry spans.
    pub word: EntryWord,
    /// First output frame of the entry within the current utterance.
    pub start_frame: usize,
    /// One past the entry's last output frame.
    pub end_frame: usize,
}

/// The noise floor of the audio fed, in dBFS.
/// `[[rr:TD-8#The runtime reports the floor]]`
///
/// The rank convention, which the record leaves open: nearest rank over the window energies
/// ascending, index `n * 5 / 100`.
struct FloorTracker {
    hop: usize,
    hop_sum_sq: VecDeque<f64>,
    open_sum_sq: f64,
    open_len: usize,
}

impl FloorTracker {
    const HOP_SECONDS: f64 = 0.05;
    const HISTORY_HOPS: usize = 200;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn new(sample_rate: f32) -> Self {
        FloorTracker {
            hop: ((sample_rate as f64 * Self::HOP_SECONDS).round() as usize).max(1),
            hop_sum_sq: VecDeque::new(),
            open_sum_sq: 0.0,
            open_len: 0,
        }
    }

    fn feed(&mut self, samples: &[f32]) {
        for &s in samples {
            self.open_sum_sq += (s as f64) * (s as f64);
            self.open_len += 1;
            if self.open_len == self.hop {
                if self.hop_sum_sq.len() == Self::HISTORY_HOPS {
                    self.hop_sum_sq.pop_front();
                }
                self.hop_sum_sq.push_back(self.open_sum_sq);
                self.open_sum_sq = 0.0;
                self.open_len = 0;
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn dbfs(&self) -> Option<f64> {
        if self.hop_sum_sq.len() < 2 {
            return None;
        }
        // A window of digital silence sorts below every level, so a floor it takes is none.
        let mut windows: Vec<Option<f64>> = self
            .hop_sum_sq
            .iter()
            .zip(self.hop_sum_sq.iter().skip(1))
            .map(|(a, b)| dbfs_of_mean_square((a + b) / (2 * self.hop) as f64))
            .collect();
        windows.sort_by(|a, b| match (a, b) {
            (Some(a), Some(b)) => a.total_cmp(b),
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
        });
        windows[windows.len() * 5 / 100]
    }
}

/// `[[rr:TD-9#Every reading carries its lead's motion]]`
#[derive(Clone, Copy, Debug, Default)]
struct Reading {
    lead_prev: Option<f32>,
    lead_now: Option<f32>,
}

impl Reading {
    fn delta(&self) -> Option<f32> {
        match (self.lead_now, self.lead_prev) {
            (Some(now), Some(prev)) => Some(now - prev),
            _ => None,
        }
    }
}

/// The history at the next advance. Returns each group's `lead_delta` in the order given,
/// with the count of groups that left the beam and the count of those that were close.
fn roll_history(
    history: &mut HashMap<Vec<Label>, Reading>,
    groups: &[(Vec<Label>, f32, Option<f32>, i32)],
) -> (Vec<Option<f32>>, (u64, u64)) {
    let mut next: HashMap<Vec<Label>, Reading> = HashMap::with_capacity(groups.len());
    let mut deltas = Vec::with_capacity(groups.len());
    for (words, _, lead, _) in groups {
        // [[rr:TD-9#Every reading carries its lead's motion]]
        let lead_prev = history.remove(words).and_then(|h| h.lead_now);
        let r = Reading {
            lead_prev,
            lead_now: *lead,
        };
        deltas.push(r.delta());
        next.insert(words.clone(), r);
    }
    // whatever the old map still holds left the beam between the two chunks
    let lost = history.len() as u64;
    let close = history
        .values()
        .filter(|h| h.lead_now.map(|l| -l < CENSUS_NATS).unwrap_or(false))
        .count() as u64;
    *history = std::mem::take(&mut next);
    (deltas, (lost, close))
}

/// `[[rr:TD-9#Every reading names its relation to the partial]]`
fn relation(words: &[Label], best: &[Label]) -> &'static str {
    if words == best {
        "same"
    } else if best.starts_with(words) {
        "prefix"
    } else if words.starts_with(best) {
        "extends"
    } else {
        "differs"
    }
}

/// Audio a recognizer keeps, one 16-bit PCM count to the unit: as 16-bit integers until a sample
/// that is not one arrives, so an integer host keeps two bytes a sample.
// [[rr:TD-13#A shared waveform path preserves scale and precision]]
enum History {
    Int(Vec<i16>),
    Float(Vec<f32>),
}

impl History {
    fn len(&self) -> usize {
        match self {
            History::Int(v) => v.len(),
            History::Float(v) => v.len(),
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::float_cmp)]
    fn extend(&mut self, samples: &[f32]) {
        if let History::Int(v) = self {
            let fits = |s: f32| s == f32::from(s as i16);
            if samples.iter().all(|&s| fits(s)) {
                v.extend(samples.iter().map(|&s| s as i16));
                return;
            }
            *self = History::Float(v.iter().map(|&s| f32::from(s)).collect());
        }
        if let History::Float(v) = self {
            v.extend_from_slice(samples);
        }
    }

    fn drop_front(&mut self, n: usize) {
        match self {
            History::Int(v) => drop(v.drain(..n)),
            History::Float(v) => drop(v.drain(..n)),
        }
    }

    fn sum_sq(&self, lo: usize, hi: usize) -> (f64, usize) {
        let sq = |s: f64| s * s;
        match self {
            History::Int(v) => v.get(lo..hi).map_or((0.0, 0), |v| {
                (v.iter().map(|&s| sq(f64::from(s))).sum(), v.len())
            }),
            History::Float(v) => v.get(lo..hi).map_or((0.0, 0), |v| {
                (v.iter().map(|&s| sq(f64::from(s))).sum(), v.len())
            }),
        }
    }

    /// Samples `lo..hi` in [-1, 1], into `out`.
    fn normalized(&self, lo: usize, hi: usize, out: &mut Vec<f32>) {
        out.clear();
        match self {
            History::Int(v) => out.extend(v[lo..hi].iter().map(|&s| f32::from(s) / FULL_SCALE)),
            History::Float(v) => out.extend(v[lo..hi].iter().map(|&s| s / FULL_SCALE)),
        }
    }

    /// The samples from `from` on, in pieces.
    fn replay(&self, from: usize, mut f: impl FnMut(&[f32])) {
        match self {
            History::Int(v) => {
                for piece in v[from..].chunks(4096) {
                    let x: Vec<f32> = piece.iter().map(|&s| f32::from(s)).collect();
                    f(&x);
                }
            }
            History::Float(v) => f(&v[from..]),
        }
    }
}

/// A stream of audio decoded against one grammar. Construct with [`new`](Self::new), feed
/// [`accept`](Self::accept), read [`partial`](Self::partial) between calls and
/// [`result`](Self::result) when [`Step::endpoint`] is set. Every result is a JSON string:
/// <https://github.com/pyscape/utter/blob/main/docs/reference/results.md>.
pub struct Recognizer<'m> {
    model: &'m Model,
    graph: Arc<VectorFst>,
    sample_rate: f32,
    state: State,
    pipeline: Option<FeaturePipeline<'m>>,
    nnet: Option<LoopedNnet<'m>>,
    decoder: Option<Decoder<'m>>,
    silence_weighting: SilenceWeighting,
    endpoint_rule: Option<crate::model::EndpointRule>,
    endpoint_veto_nats: Option<f32>,
    endpoint_floor_margin_db: Option<f32>,
    /// The rule that fired on the last `accept`, with the frame count it fired at.
    last_endpoint: Option<(usize, Endpoint)>,
    frame_offset: usize,
    samples_processed: u64,
    samples_round_start: u64,
    /// PCM fed since the pipeline began, from sample `pcm_first` on, for energy under words.
    pcm: History,
    pcm_first: usize,
    /// A call's samples in raw scale, reused from call to call.
    scratch: Vec<f32>,
    floor: FloorTracker,
    words: bool,
    partial_words: bool,
    partial_alternatives: usize,
    max_alternatives: usize,
    /// Word at each partial position and the sample position since which it has held.
    stable: Vec<(Label, u64)>,
    /// `[[rr:TD-9#Every reading carries its lead's motion]]`
    history: HashMap<Vec<Label>, Reading>,
    /// `[[rr:TD-7#Decision outcome]]`
    readings: Option<Vec<Path>>,
    readings_frames: Option<usize>,
    readings_n: usize,
    trace_groups: bool,
    group_trace: Vec<String>,
    /// `[[rr:TD-9#No lattice is added for this feature]]`
    census: bool,
    /// Groups that were in the beam at the previous chunk and are not at this one, and those of
    /// them that were within `CENSUS_NATS` of the leader when last seen.
    readings_lost: u64,
    readings_lost_close: u64,
    /// `merges_close` from decoders this recognizer has already discarded.
    merges_carried: u64,
    /// The decoder's best path without final costs, the decoded frame count it was read at,
    /// and the counts the silence weighting's frame labels and the stable list were last
    /// brought up to. `[[rr:TD-7#Decision outcome]]`
    best_path: Option<Path>,
    best_path_frames: Option<usize>,
    sw_traceback_frames: Option<usize>,
    stable_frames: Option<usize>,
    last_result: String,
    spk: Option<&'m SpeakerModel>,
    spk_stream: Option<SpeakerStream<'m>>,
    /// Evidence already pooled, by the frames it could pool and the floor threshold that
    /// chose them; cleared with each utterance.
    spk_cache: Mutex<HashMap<Vec<u64>, Option<Arc<Pooled>>>>,
    /// The stream frame the speech the best path is in at the frontier began at, if it is.
    spk_best_speech: Option<usize>,
    /// With TitaNet set. `[[rr:TD-15#The recognizer embeds a word once its span has closed]]`
    spk_spans: Option<SpanEmbedder<'m>>,
    /// `[[rr:TD-15#Readings and alternatives: decided by measurement]]`
    spk_reading_jobs: bool,
    /// `[[rr:TD-15#The floor margin: decided by measurement]]`
    spk_trim_floor: bool,
    /// `[[rr:TD-15#Embedding runs in slices between advances]]`
    spk_slice_budget: u64,
}

/// A span to embed: as reported, the key its evidence is kept by, and the samples embedded, both
/// on the stream's sample clock.
#[derive(Clone, Copy)]
struct SpanJob {
    span: (u64, u64),
    embed: (u64, u64),
}

/// TitaNet's embeddings of the spans of the current utterance.
struct SpanEmbedder<'m> {
    model: &'m TitaNet,
    queued: HashSet<(u64, u64)>,
    queue: VecDeque<SpanJob>,
    running: Option<SpanJob>,
    job: Embedding,
    samples: Vec<f32>,
    done: HashMap<(u64, u64), Pooled>,
}

impl<'m> SpanEmbedder<'m> {
    fn new(model: &'m TitaNet) -> Self {
        SpanEmbedder {
            model,
            queued: HashSet::new(),
            queue: VecDeque::new(),
            running: None,
            job: Embedding::new(),
            samples: Vec::new(),
            done: HashMap::new(),
        }
    }

    fn forget(&mut self) {
        self.queued.clear();
        self.queue.clear();
        self.running = None;
        self.done.clear();
    }
}

/// Work units a call that does not advance the decoder spends embedding.
/// `[[rr:TD-15#What a block may cost]]`
pub const SPK_SLICE_BUDGET: u64 = 50_000_000;

/// Evidence as pooled once, written as JSON once for every result that carries it.
struct Pooled {
    vector_json: String,
    span_json: String,
}

// The look-ahead's settings, and what they cost and save:
// [[rr:TD-14#Mean normalisation looks back only]]
/// A frame this far over the floor is likely speech, so the look-ahead computes its row.
const LOOK_AHEAD_OVER_FLOOR_DB: f64 = 20.0;
/// The look-ahead's level before a floor exists.
const LOOK_AHEAD_DBFS: f64 = -40.0;
/// Frames either side of a likely-speech frame the look-ahead also computes, since a word's
/// span starts and ends in quieter frames than its loudest.
const LOOK_AHEAD_PAD: usize = 16;
/// Readings the look-ahead follows when the host asks for none.
const LOOK_AHEAD_READINGS: usize = 3;
/// A reading's cost over the best's within which the look-ahead follows it.
const LOOK_AHEAD_LEAD: f32 = 10.0;
/// Rows of the best path's speech the look-ahead waits for between advances.
const LOOK_AHEAD_BATCH: usize = 20;

fn escape_json_number(v: f64) -> String {
    if v.is_finite() {
        let s = format!("{v:.6}");
        let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
        if s.is_empty() || s == "-" || s == "-0" {
            "0".into()
        } else {
            s
        }
    } else if v > 0.0 {
        "1e999".into()
    } else {
        "-1e999".into()
    }
}

fn number_or_null(v: Option<f64>) -> String {
    match v {
        Some(x) => escape_json_number(x),
        None => "null".into(),
    }
}

/// Full scale in the front end's units, one 16-bit PCM count to the unit: a normalized sample
/// of 1.0, and 0 dBFS.
const FULL_SCALE: f32 = 32768.0;

/// Digital silence has no level in decibels, so it reads as no level rather than as a number
/// every threshold sits above.
fn dbfs_of_mean_square(mean_square: f64) -> Option<f64> {
    let rms = mean_square.sqrt();
    (rms > 0.0).then(|| 20.0 * (rms / f64::from(FULL_SCALE)).log10())
}

/// Why [`Recognizer::accept_f32`] refused a block. Nothing of the block was accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct AudioInputError {
    /// Index within the block of the first sample refused.
    pub index: usize,
    /// What is wrong with it.
    pub kind: AudioInputErrorKind,
}

/// What is wrong with a sample [`Recognizer::accept_f32`] refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AudioInputErrorKind {
    /// NaN.
    NotANumber,
    /// Positive or negative infinity.
    Infinite,
    /// Finite and outside full scale, `[-1.0, 1.0]`.
    OutOfRange,
}

impl std::fmt::Display for AudioInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self.kind {
            AudioInputErrorKind::NotANumber => "is not a number",
            AudioInputErrorKind::Infinite => "is infinite",
            AudioInputErrorKind::OutOfRange => "is outside [-1.0, 1.0]",
        };
        write!(f, "sample {} of the block {what}", self.index)
    }
}

impl std::error::Error for AudioInputError {}

fn check_normalized(samples: &[f32]) -> Result<(), AudioInputError> {
    let Some(index) = samples.iter().position(|x| !(-1.0..=1.0).contains(x)) else {
        return Ok(());
    };
    let x = samples[index];
    let kind = if x.is_nan() {
        AudioInputErrorKind::NotANumber
    } else if x.is_infinite() {
        AudioInputErrorKind::Infinite
    } else {
        AudioInputErrorKind::OutOfRange
    };
    Err(AudioInputError { index, kind })
}

impl<'m> Recognizer<'m> {
    /// Compile the grammar and prepare a stream. An empty grammar is refused: decoding against
    /// the full language model is out of scope.
    pub fn new(model: &'m Model, sample_rate: f32, grammar: &[String]) -> std::io::Result<Self> {
        Self::with_options(model, sample_rate, grammar, &RecognizerOptions::default())
    }

    /// [`new`](Self::new) with [`RecognizerOptions`]. Fails on an empty grammar, one over
    /// [`MAX_GRAMMAR_WORDS`] distinct words, or a composed graph over
    /// [`max_graph_states`](RecognizerOptions::max_graph_states).
    pub fn with_options(
        model: &'m Model,
        sample_rate: f32,
        grammar: &[String],
        options: &RecognizerOptions,
    ) -> std::io::Result<Self> {
        let max_states = options.max_graph_states;
        if grammar.is_empty() {
            return Err(crate::kaldi_io::err("a grammar is required"));
        }
        let distinct: std::collections::HashSet<&str> = grammar
            .iter()
            .flat_map(|s| s.split(' '))
            .filter(|w| !w.is_empty())
            .collect();
        if distinct.len() >= MAX_GRAMMAR_WORDS {
            return Err(crate::kaldi_io::err(&format!(
                "grammar has {} distinct words; fewer than {MAX_GRAMMAR_WORDS} are supported",
                distinct.len()
            )));
        }
        let graph = model.grammar_graph(grammar, options.unknown_cost, max_states, |w| {
            eprintln!("utter: {w}")
        })?;
        let sw = SilenceWeighting::new(
            &model.conf.silence_phones,
            options.silence_weight,
            model.conf.frame_subsampling_factor,
        );
        Ok(Recognizer {
            silence_weighting: sw,
            endpoint_rule: options.endpoint_rule.clone(),
            endpoint_veto_nats: None,
            endpoint_floor_margin_db: None,
            last_endpoint: None,
            model,
            graph,
            sample_rate,
            state: State::Initialized,
            pipeline: None,
            nnet: None,
            decoder: None,
            frame_offset: 0,
            samples_processed: 0,
            samples_round_start: 0,
            pcm: History::Int(Vec::new()),
            pcm_first: 0,
            scratch: Vec::new(),
            floor: FloorTracker::new(sample_rate),
            words: false,
            partial_words: false,
            partial_alternatives: 0,
            max_alternatives: 0,
            stable: Vec::new(),
            history: HashMap::new(),
            readings: None,
            readings_frames: None,
            readings_n: 0,
            trace_groups: false,
            group_trace: Vec::new(),
            census: false,
            readings_lost: 0,
            readings_lost_close: 0,
            merges_carried: 0,
            best_path: None,
            best_path_frames: None,
            sw_traceback_frames: None,
            stable_frames: None,
            last_result: String::new(),
            spk: None,
            spk_stream: None,
            spk_cache: Mutex::new(HashMap::new()),
            spk_best_speech: None,
            spk_spans: None,
            spk_reading_jobs: false,
            spk_trim_floor: false,
            spk_slice_budget: SPK_SLICE_BUDGET,
        })
    }

    /// The compiled decoding graph, shared with any other recognizer on the same grammar.
    #[doc(hidden)]
    pub fn graph(&self) -> &VectorFst {
        &self.graph
    }
    /// The MFCC frames the current pipeline keeps and the frame the first of them is, for tests
    /// that compare two ways of feeding it.
    #[doc(hidden)]
    pub fn features(&self) -> (&[Vec<f32>], usize) {
        match self.pipeline.as_ref() {
            Some(p) => p.mfcc.frames(),
            None => (&[], 0),
        }
    }
    /// libvosk's `SetWords`: word entries on finals.
    pub fn set_words(&mut self, on: bool) {
        self.words = on;
    }
    /// libvosk's `SetPartialWords`: word entries on partials and on each reading. The partial
    /// stays the best path; it does not switch to a lattice as libvosk's does.
    pub fn set_partial_words(&mut self, on: bool) {
        self.partial_words = on;
    }
    /// The host's endpoint bound, [`RecognizerOptions::endpoint_rule`]: a final once the trailing
    /// silence reaches `trailing_ms`, unless `extending_veto_nats` is set and a reading that
    /// extends the partial by a further word is within that many nats of it. The span cannot see
    /// a word beginning, since the word's label is not yet on the best path while the silence
    /// before it still counts; the beam can, and the veto reads it. `None` removes the bound.
    pub fn set_endpoint_bound(
        &mut self,
        trailing_ms: Option<f32>,
        extending_veto_nats: Option<f32>,
    ) {
        self.endpoint_rule = trailing_ms.map(|ms| crate::model::EndpointRule {
            must_contain_nonsilence: true,
            min_trailing_silence: ms / 1000.0,
            max_relative_cost: f32::INFINITY,
            min_utterance_length: 0.0,
        });
        self.endpoint_veto_nats = extending_veto_nats;
    }
    /// A margin in dB over the reported floor within which the bound reads a wordless path as
    /// silence: a trailing `[speech]` entry whose energy is within it counts toward the bound's
    /// trailing silence, and a reading extending the partial by a word whose energy is within it
    /// does not veto. Either span must be at least as long as the bound before its energy is
    /// read, so a quiet onset still forming is not taken for hiss. A final the bound reaches this way is the path as it stood, no word forced
    /// onto it, and names `floor` as its endpoint. `None` removes the margin; the bound is then as
    /// [`set_endpoint_bound`](Self::set_endpoint_bound) alone describes it. `[[rr:TD-12#Decision outcome]]`
    pub fn set_endpoint_floor_margin(&mut self, db: Option<f32>) {
        self.endpoint_floor_margin_db = db;
    }

    /// Off by default; it never touches a decision.
    #[doc(hidden)]
    pub fn set_census(&mut self, on: bool) {
        self.census = on;
        if let Some(d) = self.decoder.as_mut() {
            d.census = on;
        }
    }

    /// Local close collisions, readings lost between chunks, and those of them that were
    /// within `CENSUS_NATS` of the leader when last seen.
    #[doc(hidden)]
    pub fn census_counts(&self) -> (u64, u64, u64) {
        let merges =
            self.merges_carried + self.decoder.as_ref().map(|d| d.merges_close).unwrap_or(0);
        (merges, self.readings_lost, self.readings_lost_close)
    }

    /// Record, per chunk, the readings the decoder carries and how each moved against the
    /// leader, for [`take_group_trace`](Self::take_group_trace). Off by default.
    #[doc(hidden)]
    pub fn set_trace_groups(&mut self, on: bool) {
        self.trace_groups = on;
    }

    /// One JSON object per chunk decoded.
    #[doc(hidden)]
    pub fn take_group_trace(&mut self) -> Vec<String> {
        std::mem::take(&mut self.group_trace)
    }

    /// Readings to carry beside every partial, 0 for none. Not in libvosk.
    /// Keys: <https://github.com/pyscape/utter/blob/main/docs/reference/results.md#a-reading>.
    pub fn set_alternatives(&mut self, n: usize) {
        self.partial_alternatives = n;
    }
    /// Alternatives on finals, libvosk's `SetMaxAlternatives`; 0 keeps the plain shape.
    pub fn set_max_alternatives(&mut self, n: usize) {
        self.max_alternatives = n;
    }

    /// libvosk's `SetSpkModel`: speaker evidence on every partial and final, and on every entry
    /// of their word lists. The current utterance's audio already fed is included. Fails when the
    /// audio's rate is below the speaker model's, or its frames are not the decoder's 10 ms.
    /// `None` removes it. With TitaNet, only word entries carry evidence, once their closed span is
    /// embedded, and the audio must be 16 kHz or faster. Keys: <https://github.com/pyscape/utter/blob/main/docs/reference/results.md#speaker-evidence>.
    // [[rr:TD-14#Decision outcome]]
    // [[rr:TD-15#Decision outcome]]
    pub fn set_spk_model(&mut self, spk: Option<&'m SpeakerModel>) -> std::io::Result<()> {
        self.spk_cache().clear();
        let Some(model) = spk else {
            self.spk = None;
            self.spk_stream = None;
            self.spk_spans = None;
            return Ok(());
        };
        if let Some(net) = model.titanet() {
            return self.set_titanet(model, net);
        }
        self.spk_spans = None;
        let xvector = model.xvector().expect("TitaNet returned above");
        if xvector.frame_shift_ms() != self.model.mfcc_opts.frame_shift_ms {
            return Err(crate::kaldi_io::err(
                "the speaker model's frames are not the decoder's",
            ));
        }
        // A final already read has ended its utterance, though its audio goes at the next accept.
        let utterance = match (self.state, self.decoder.as_ref()) {
            (State::Endpoint, Some(d)) => self.frame_offset + d.num_frames_decoded(),
            _ => self.frame_offset,
        };
        let from = self.pcm_kept_from(utterance).max(self.pcm_first);
        let (hop, _) = self.spk_hop_and_window();
        let mut stream = SpeakerStream::starting_at(model, self.sample_rate, from / hop)?;
        if self.pipeline.is_some() {
            self.pcm.replay(from - self.pcm_first, |s| stream.accept(s));
            stream.forget_before(self.frame_offset * self.model.conf.frame_subsampling_factor);
        }
        self.spk = Some(model);
        self.spk_stream = self.pipeline.is_some().then_some(stream);
        Ok(())
    }

    // [[rr:TD-15#The recognizer embeds a word once its span has closed]]
    fn set_titanet(&mut self, model: &'m SpeakerModel, net: &'m TitaNet) -> std::io::Result<()> {
        let want = 16_000.0;
        if !(self.sample_rate == want
            || (self.sample_rate > want && self.sample_rate.fract() == 0.0))
        {
            return Err(crate::kaldi_io::err(&format!(
                "TitaNet needs audio at {want} Hz or faster, in whole hertz"
            )));
        }
        self.spk_stream = None;
        let same = self.spk.is_some_and(|m| std::ptr::eq(m, model)) && self.spk_spans.is_some();
        self.spk = Some(model);
        if !same {
            self.spk_spans = Some(SpanEmbedder::new(net));
        }
        if self.state == State::Running {
            self.refresh_best_path();
            self.spk_queue_closed_paths();
        }
        Ok(())
    }

    /// Which rule readings follow with TitaNet set: off, they carry only spans the best path
    /// queued; on, their closed words are queued after the best path's. Final alternatives exist
    /// only at a final, where no embedding runs, so under either rule they carry what is embedded.
    #[doc(hidden)]
    // [[rr:TD-15#Readings and alternatives: decided by measurement]]
    pub fn set_spk_reading_jobs(&mut self, on: bool) {
        self.spk_reading_jobs = on;
    }

    /// Which rule a floor margin follows with TitaNet set: off, a span is embedded as reported;
    /// on, its frames within the margin of the floor at either edge are cut first.
    #[doc(hidden)]
    // [[rr:TD-15#The floor margin: decided by measurement]]
    pub fn set_spk_trim_floor(&mut self, on: bool) {
        self.spk_trim_floor = on;
    }

    /// Work units per slice, [`SPK_SLICE_BUDGET`] unless set; for tests that vary the slices.
    #[doc(hidden)]
    pub fn set_spk_slice_budget(&mut self, units: u64) {
        self.spk_slice_budget = units.max(1);
    }

    fn decoder_config(&self) -> DecoderConfig {
        let c = &self.model.conf;
        DecoderConfig {
            beam: c.beam,
            max_active: c.max_active,
            min_active: c.min_active,
            beam_delta: 0.5,
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn frame_samples(&self) -> u64 {
        // one output frame is subsampling x 10 ms
        (self.model.conf.frame_subsampling_factor as f64
            * self.model.mfcc_opts.frame_shift_ms as f64
            * 0.001
            * self.sample_rate as f64)
            .round() as u64
    }

    fn rebuild(&mut self) {
        if let Some(d) = self.decoder.as_ref() {
            self.merges_carried += d.merges_close;
        }
        self.samples_round_start += self.samples_processed;
        self.samples_processed = 0;
        self.frame_offset = 0;
        self.pcm = History::Int(Vec::new());
        self.pcm_first = 0;
        self.forget_best_path();
        let mut opts = self.model.mfcc_opts.clone();
        opts.sample_rate = self.sample_rate;
        self.pipeline = Some(FeaturePipeline::new(&opts, self.model.ivector.as_ref()));
        self.nnet = Some(LoopedNnet::new(self.model));
        let cfg = self.decoder_config();
        let mut dec = Decoder::new(
            self.graph.clone(),
            &self.model.tm.tid2pdf,
            &self.model.tm.tid2phone,
            cfg,
        );
        dec.census = self.census;
        self.decoder = Some(dec);
        self.stable.clear();
        self.spk_stream = self
            .spk
            .filter(|m| m.xvector().is_some())
            .and_then(|m| SpeakerStream::new(m, self.sample_rate).ok());
        self.spk_cache().clear();
        if let Some(sp) = self.spk_spans.as_mut() {
            sp.forget();
        }
    }

    /// libvosk's `CleanUp`: a new utterance on the same pipeline, or a new pipeline after a
    /// final result or 20,000 frames.
    fn clean_up(&mut self) {
        self.forget_best_path();
        self.silence_weighting = SilenceWeighting::new(
            &self.model.conf.silence_phones,
            SILENCE_WEIGHT,
            self.model.conf.frame_subsampling_factor,
        );
        if let Some(d) = &self.decoder {
            self.frame_offset += d.num_frames_decoded();
        }
        match (self.decoder.as_mut(), self.nnet.as_mut()) {
            (Some(dec), Some(nnet))
                if self.state != State::Finalized && self.frame_offset <= 20000 =>
            {
                nnet.set_frame_offset(self.frame_offset);
                dec.init_decoding();
                self.stable.clear();
                let first = self.frame_offset * self.model.conf.frame_subsampling_factor;
                if let Some(s) = self.spk_stream.as_mut() {
                    s.forget_before(first);
                }
                if let Some(iv) = self.pipeline.as_mut().and_then(|p| p.ivector.as_mut()) {
                    iv.forget_before(first);
                }
                self.forget_pcm();
                self.spk_cache().clear();
                if let Some(sp) = self.spk_spans.as_mut() {
                    sp.forget();
                }
            }
            _ => self.rebuild(),
        }
    }

    /// The first sample of the pipeline kept for the utterance that begins at decoder frame
    /// `utterance`: its first stream frame's start, or its first sample if earlier, back to a
    /// stream frame's start so that a speaker model set later can start from it.
    // [[rr:TD-14#A model set mid-stream starts at the current utterance]]
    #[allow(clippy::cast_possible_truncation)]
    fn pcm_kept_from(&self, utterance: usize) -> usize {
        let (hop, _) = self.spk_hop_and_window();
        let frame = utterance * self.model.conf.frame_subsampling_factor;
        let sample = utterance * self.frame_samples() as usize;
        (frame * hop).min(sample) / hop * hop
    }

    /// Drop the audio before the current utterance.
    fn forget_pcm(&mut self) {
        let keep = self.pcm_kept_from(self.frame_offset);
        let n = keep.saturating_sub(self.pcm_first).min(self.pcm.len());
        self.pcm.drop_front(n);
        self.pcm_first += n;
    }

    /// The sum of squares and the count of the kept audio from sample `lo` of the pipeline to
    /// `hi`, or to the last kept.
    fn pcm_sum_sq(&self, lo: usize, hi: usize) -> (f64, usize) {
        assert!(
            lo >= self.pcm_first,
            "audio asked for before the audio kept"
        );
        let hi = hi.min(self.pcm_first + self.pcm.len());
        self.pcm
            .sum_sq(lo - self.pcm_first, hi.saturating_sub(self.pcm_first))
    }

    fn forget_best_path(&mut self) {
        self.spk_best_speech = None;
        self.forget_readings();
        self.best_path = None;
        self.best_path_frames = None;
        self.sw_traceback_frames = None;
        self.stable_frames = None;
    }

    fn refresh_best_path(&mut self) {
        let Some(dec) = self.decoder.as_ref() else {
            return;
        };
        let decoded = dec.num_frames_decoded();
        if self.best_path_frames == Some(decoded) {
            return;
        }
        self.best_path = dec.best_path(false);
        self.best_path_frames = Some(decoded);
    }

    /// libvosk's `UpdateSilenceWeights`, run before every decoding advance.
    fn update_silence_weights(&mut self) {
        let sf = self.model.conf.frame_subsampling_factor;
        let (Some(pipe), Some(dec)) = (self.pipeline.as_ref(), self.decoder.as_ref()) else {
            return;
        };
        if pipe.ivector.is_none() {
            return;
        }
        let ready = pipe.mfcc.num_frames_ready();
        if !self.silence_weighting.active() || ready == 0 {
            return;
        }
        let decoded = dec.num_frames_decoded();
        self.refresh_best_path();
        if self.sw_traceback_frames != Some(decoded) {
            if let Some(path) = self.best_path.as_ref() {
                self.silence_weighting
                    .compute_current_traceback(path, decoded);
            }
            self.sw_traceback_frames = Some(decoded);
        }
        let deltas = self
            .silence_weighting
            .get_delta_weights(ready, self.frame_offset * sf);
        let iv = self.pipeline.as_mut().unwrap().ivector.as_mut().unwrap();
        iv.update_frame_weights(&deltas);
    }

    /// `[[rr:TD-2#The network]]`
    fn chunk_frames(&self) -> usize {
        (self.model.conf.frames_per_chunk / self.model.conf.frame_subsampling_factor).max(1)
    }

    /// The drain stops at each chunk boundary, whatever the block size that fed it.
    /// `[[rr:TD-9#Readings are read once per decoding advance]]`
    fn advance_decoding(&mut self) {
        let chunk = self.chunk_frames();
        loop {
            {
                let (Some(pipe), Some(nnet), Some(dec)) = (
                    self.pipeline.as_mut(),
                    self.nnet.as_mut(),
                    self.decoder.as_mut(),
                ) else {
                    return;
                };
                let ready = nnet.num_frames_ready(pipe);
                if dec.num_frames_decoded() >= ready {
                    return;
                }
                let offset = nnet.frame_offset();
                let boundary = (dec.num_frames_decoded() + offset) / chunk * chunk + chunk;
                let limit = boundary.saturating_sub(offset).min(ready);
                while dec.num_frames_decoded() < limit {
                    let row = nnet.frame(pipe, dec.num_frames_decoded());
                    dec.advance_frame(row);
                }
            }
            self.update_readings();
        }
    }

    /// `[[rr:TD-9#Readings are read once per decoding advance]]`
    fn update_readings(&mut self) {
        if self.partial_alternatives == 0 && !self.trace_groups && !self.census {
            return;
        }
        let n = self.partial_alternatives;
        let (decoded, sample, groups, paths) = {
            let Some(dec) = self.decoder.as_ref() else {
                return;
            };
            let decoded = dec.num_frames_decoded();
            if self.readings_frames == Some(decoded) {
                return;
            }
            let groups = dec.grouped(false);
            // Without final costs the rank is the cost, so the best other cost is the first
            // group's, or the second's for the first group itself.
            let leads: Vec<(Vec<Label>, f32, Option<f32>, i32)> = groups
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    let lead = match (i, groups.len()) {
                        (_, 0 | 1) => None,
                        (0, _) => Some(groups[1].cost - g.cost),
                        _ => Some(groups[0].cost - g.cost),
                    };
                    (g.words.clone(), g.cost, lead, g.token.phone)
                })
                .collect();
            let paths = dec.trace_groups(&groups, n);
            (decoded, self.sample_of(decoded), leads, paths)
        };
        let (deltas, (lost, close)) = roll_history(&mut self.history, &groups);
        self.readings_lost += lost;
        self.readings_lost_close += close;
        if self.trace_groups {
            self.push_group_trace(decoded, sample, &groups, &deltas);
        }
        self.readings = Some(paths);
        self.readings_frames = Some(decoded);
        self.readings_n = n;
    }

    /// `[[rr:TD-9#Readings are read once per decoding advance]]`
    fn refresh_readings(&mut self) {
        let n = self.partial_alternatives;
        let Some(dec) = self.decoder.as_ref() else {
            return;
        };
        let decoded = dec.num_frames_decoded();
        if self.readings_frames == Some(decoded) && self.readings_n == n {
            return;
        }
        let groups = dec.grouped(false);
        let paths = dec.trace_groups(&groups, n);
        self.readings = Some(paths);
        self.readings_frames = Some(decoded);
        self.readings_n = n;
    }

    fn forget_readings(&mut self) {
        self.history.clear();
        self.readings = None;
        self.readings_frames = None;
        self.readings_n = 0;
    }

    /// `stream --trace-groups`: every surviving group at every chunk, the tracker's oracle.
    fn push_group_trace(
        &mut self,
        decoded: usize,
        sample: u64,
        groups: &[(Vec<Label>, f32, Option<f32>, i32)],
        deltas: &[Option<f32>],
    ) {
        let best: &[Label] = groups.first().map(|g| g.0.as_slice()).unwrap_or(&[]);
        let mut out = format!(
            "{{\"frames\": {}, \"sample\": {}, \"groups\": [",
            self.frame_offset + decoded,
            sample
        );
        for (i, (words, cost, lead, phone)) in groups.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str("{\"text\": ");
            write_string(&mut out, &self.text_of(words, Some(*phone)));
            out.push_str(&format!(
                ", \"confidence\": {}, \"relation\": \"{}\", \"lead\": {}, \"lead_delta\": {}}}",
                escape_json_number(-(*cost as f64)),
                relation(words, best),
                number_or_null(lead.map(f64::from)),
                number_or_null(deltas[i].map(f64::from)),
            ));
        }
        out.push_str("]}");
        self.group_trace.push(out);
    }

    /// libvosk's `AcceptWaveform`: feed 16-bit mono PCM at the recognizer's rate. Any block
    /// size; the decoder advances in 200 ms chunks internally and the partial is current to the
    /// last decoded frame.
    pub fn accept(&mut self, samples: &[i16]) -> Step {
        self.accept_raw(samples, f32::from)
    }

    /// [`accept`](Self::accept) for normalized samples: mono PCM at the recognizer's rate, full
    /// scale at -1.0 and 1.0, and every sample finite and within it. A block holding any other
    /// value is refused whole and leaves the recognizer as it was. The block is read during the
    /// call and not kept. A 16-bit sample `s` is `s as f32 / 32768.0` exactly, and gives what
    /// [`accept`](Self::accept) gives for `s`; the two may be interleaved on one recognizer.
    // [[rr:TD-13#The connection carries normalized waveform samples]]
    pub fn accept_f32(&mut self, samples: &[f32]) -> Result<Step, AudioInputError> {
        check_normalized(samples)?;
        Ok(self.accept_raw(samples, |x| x * FULL_SCALE))
    }

    // [[rr:TD-13#A shared waveform path preserves scale and precision]]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn accept_raw<S: Copy>(&mut self, samples: &[S], raw: impl Fn(S) -> f32) -> Step {
        if !(self.state == State::Running || self.state == State::Initialized) {
            self.clean_up();
        } else if self.pipeline.is_none() {
            self.rebuild();
        }
        self.state = State::Running;
        let decoded_before = self.decoder.as_ref().map(|d| d.num_frames_decoded());
        let step = (self.sample_rate * 0.2) as usize;
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        scratch.extend(samples.iter().map(|&s| raw(s)));
        for piece in scratch.chunks(step.max(1)) {
            self.pipeline.as_mut().unwrap().accept(piece);
            self.update_silence_weights();
            self.advance_decoding();
        }
        self.pcm.extend(&scratch);
        let advanced = self.decoder.as_ref().map(|d| d.num_frames_decoded()) != decoded_before;
        if let Some(s) = self.spk_stream.as_mut() {
            s.accept(&scratch);
            if advanced {
                s.catch_up();
            }
        }
        self.floor.feed(&scratch);
        self.scratch = scratch;
        self.samples_processed += samples.len() as u64;
        self.update_stable();
        if !advanced {
            self.spk_look_ahead(samples.len());
            self.spk_slices();
        }
        let reason = self.endpoint_reason();
        let endpoint = reason.is_some();
        let decoded = self
            .decoder
            .as_ref()
            .map(|d| d.num_frames_decoded())
            .unwrap_or(0);
        if let Some(r) = reason {
            self.last_endpoint = Some((decoded, r));
        }
        let sample =
            self.samples_round_start + (self.frame_offset + decoded) as u64 * self.frame_samples();
        Step { endpoint, sample }
    }

    /// libvosk's `EndpointDetected` over the current decoding.
    pub fn endpoint_detected(&self) -> bool {
        self.endpoint_reason().is_some()
    }

    /// The first of the model's rules that fires, else the host's bound if it does.
    // [[rr:TD-11#Decision outcome]]
    #[allow(clippy::cast_precision_loss)]
    pub fn endpoint_reason(&self) -> Option<Endpoint> {
        let dec = self.decoder.as_ref()?;
        let n = dec.num_frames_decoded();
        if n == 0 {
            return None;
        }
        let conf = &self.model.conf;
        let shift =
            conf.frame_subsampling_factor as f32 * self.model.mfcc_opts.frame_shift_ms * 0.001;
        let fresh;
        let path = if self.best_path_frames == Some(n) {
            self.best_path.as_ref()
        } else {
            fresh = dec.best_path(false);
            fresh.as_ref()
        };
        let trailing = path
            .map(|p| p.trailing_silence_frames(&conf.silence_phones))
            .unwrap_or(0) as f32
            * shift;
        let utterance = n as f32 * shift;
        let relative = dec.final_relative_cost();
        let contains_nonsilence = utterance > trailing;
        let fires = |r: &crate::model::EndpointRule, trailing: f32| {
            (contains_nonsilence || !r.must_contain_nonsilence)
                && trailing >= r.min_trailing_silence
                && relative <= r.max_relative_cost
                && utterance >= r.min_utterance_length
        };
        if let Some(i) = conf.rules.iter().position(|r| fires(r, trailing)) {
            return Some(Endpoint::Rule(
                u8::try_from(i).expect("rule index fits a byte") + 1,
            ));
        }
        let bound = self.endpoint_rule.as_ref()?;
        // [[rr:TD-12#Decision outcome]]
        let floor_margin = match (self.endpoint_floor_margin_db, self.floor.dbfs()) {
            (Some(margin), Some(floor)) => Some((floor, f64::from(margin))),
            _ => None,
        };
        // A span shorter than the bound is an onset still forming, whose energy says nothing
        // yet; the margin judges no span shorter than the bound itself.
        let at_floor = |start_frame: usize, end_frame: usize| {
            floor_margin.is_some_and(|(floor, margin)| {
                (end_frame - start_frame) as f32 * shift >= bound.min_trailing_silence
                    && self
                        .energy_dbfs(self.sample_of(start_frame), self.sample_of(end_frame))
                        .is_some_and(|e| e <= floor + margin)
            })
        };
        let reason = if fires(bound, trailing) {
            Endpoint::Bound
        } else if floor_margin.is_some()
            && path.is_some_and(|p| {
                fires(
                    bound,
                    self.trailing_wordless_frames(p, n, at_floor) as f32 * shift,
                )
            })
        {
            Endpoint::Floor
        } else {
            return None;
        };
        let vetoed = match self.endpoint_veto_nats {
            None => false,
            Some(nats) => {
                let alts = dec.alternatives(false, usize::MAX);
                let Some(top) = alts.first() else {
                    return Some(reason);
                };
                alts.iter().skip(1).any(|a| {
                    a.words.len() > top.words.len()
                        && a.words[..top.words.len()] == top.words[..]
                        && a.cost - top.cost <= nats
                        && !self
                            .align(a)
                            .get(top.words.len())
                            .is_some_and(|w| at_floor(w.start_frame, w.end_frame))
                })
            }
        };
        (!vetoed).then_some(reason)
    }

    /// Frames of the wordless span that closes a path: the `[sil]` entries and a `[speech]`
    /// entry, contiguous back from the last decoded frame, the `[speech]` entry only where
    /// `at_floor` says so of its span. Zero as soon as a word, a gap or a `[speech]` entry above
    /// the floor is met.
    fn trailing_wordless_frames(
        &self,
        path: &Path,
        n: usize,
        at_floor: impl Fn(usize, usize) -> bool,
    ) -> usize {
        let mut start = n;
        for e in self.entries(path).iter().rev() {
            if e.end_frame != start {
                break;
            }
            match e.word {
                EntryWord::Word(_) => break,
                EntryWord::Speech if !at_floor(e.start_frame, e.end_frame) => break,
                EntryWord::Speech | EntryWord::Silence => start = e.start_frame,
            }
        }
        n - start
    }

    /// Milliseconds each entry of a path's word list has held its place in the partial, read off
    /// the stable list by position: zero where the list holds something else, which is a word
    /// the final carries and no partial showed. `[[rr:TD-11#Decision outcome]]`
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn holds(&self, path: &Path, now: u64) -> Vec<(Entry, u64)> {
        self.entries(path)
            .into_iter()
            .enumerate()
            .map(|(j, e)| {
                let token = match e.word {
                    EntryWord::Word(w) => w,
                    EntryWord::Silence => -1,
                    EntryWord::Speech => -2,
                };
                let since = match self.stable.get(j) {
                    Some(&(t, since)) if t == token => since,
                    _ => now,
                };
                let ms = ((now - since) as f64 / self.sample_rate as f64 * 1000.0).round() as u64;
                (e, ms)
            })
            .collect()
    }

    fn write_final_words(&self, out: &mut String, path: &Path, now: u64) {
        let mut first = true;
        for (span, hold) in self.holds(path, now) {
            if !matches!(span.word, EntryWord::Word(_)) {
                continue;
            }
            if !first {
                out.push_str(", ");
            }
            first = false;
            // [[rr:TD-8#A final word carries its energy]]
            self.write_word(out, &span, true, true, now);
            out.pop();
            out.push_str(&format!(", \"stable_ms\": {hold}"));
            self.push_entry_spk(out, &span);
            out.push('}');
        }
    }

    fn update_stable(&mut self) {
        let Some(dec) = self.decoder.as_ref() else {
            return;
        };
        let decoded = dec.num_frames_decoded();
        if self.stable_frames == Some(decoded) {
            return;
        }
        self.stable_frames = Some(decoded);
        self.refresh_best_path();
        let now = self.samples_round_start + self.samples_processed;
        let entries = self.best_path.as_ref().map(|path| self.entries(path));
        let sf = self.model.conf.frame_subsampling_factor;
        self.spk_best_speech = entries
            .as_ref()
            .and_then(|es| es.last())
            .filter(|e| e.word != EntryWord::Silence && e.end_frame >= decoded)
            .map(|e| (self.frame_offset + e.start_frame) * sf);
        let Some(entries) = entries else {
            return;
        };
        self.spk_queue_closed_paths();
        // [[rr:TD-14#Mean normalisation looks back only]]
        if let Some(stream) = self.spk_stream.as_ref() {
            let speech: Vec<(usize, usize)> = entries
                .iter()
                .filter(|e| e.word != EntryWord::Silence)
                .map(|e| (e.start_frame, e.end_frame))
                .collect();
            let frames = self.spk_frames(&speech);
            if stream.poolable(&frames) >= crate::speaker::MIN_FRAMES {
                let _computed = stream.compute(&frames);
                #[cfg(test)]
                tests::ROWS.with(|r| r.set((r.get().0, r.get().1 + _computed)));
            }
        }
        let tokens: Vec<Label> = entries
            .iter()
            .map(|e| match e.word {
                EntryWord::Word(w) => w,
                EntryWord::Silence => -1,
                EntryWord::Speech => -2,
            })
            .collect();
        let mut keep = 0;
        while keep < tokens.len() && keep < self.stable.len() && self.stable[keep].0 == tokens[keep]
        {
            keep += 1;
        }
        self.stable.truncate(keep);
        for &w in &tokens[keep..] {
            self.stable.push((w, now));
        }
    }

    /// Align a path's words to its phone segments through `word_boundary.int`.
    #[doc(hidden)]
    pub fn align(&self, path: &Path) -> Vec<WordSpan> {
        let wb = &self.model.word_boundary;
        let mut spans = Vec::new();
        let mut open: Option<usize> = None;
        let mut next_word = 0;
        let push = |start: usize, end: usize, next_word: &mut usize, spans: &mut Vec<WordSpan>| {
            if *next_word < path.words.len() {
                spans.push(WordSpan {
                    word: path.words[*next_word],
                    start_frame: start,
                    end_frame: end,
                });
                *next_word += 1;
            }
        };
        for seg in &path.phones {
            match wb.get(&seg.phone).copied().unwrap_or(WordBoundary::Nonword) {
                WordBoundary::Nonword => {
                    if let Some(start) = open.take() {
                        push(start, seg.start, &mut next_word, &mut spans);
                    }
                }
                WordBoundary::Begin => {
                    if let Some(start) = open.take() {
                        push(start, seg.start, &mut next_word, &mut spans);
                    }
                    open = Some(seg.start);
                }
                WordBoundary::Internal => {
                    if open.is_none() {
                        open = Some(seg.start);
                    }
                }
                WordBoundary::End => {
                    let start = open.take().unwrap_or(seg.start);
                    push(start, seg.end, &mut next_word, &mut spans);
                }
                WordBoundary::Singleton => {
                    if let Some(start) = open.take() {
                        push(start, seg.start, &mut next_word, &mut spans);
                    }
                    push(seg.start, seg.end, &mut next_word, &mut spans);
                }
            }
        }
        if let Some(start) = open {
            let end = path.phones.last().map(|s| s.end).unwrap_or(start);
            push(start, end, &mut next_word, &mut spans);
        }
        // Words whose phones have not closed yet, or that the alignment could not place, keep
        // the frame they were emitted on.
        while next_word < path.words.len() {
            let f = path.word_frames[next_word];
            spans.push(WordSpan {
                word: path.words[next_word],
                start_frame: f,
                end_frame: f + 1,
            });
            next_word += 1;
        }
        spans
    }

    #[allow(clippy::cast_precision_loss)]
    fn seconds(&self, frame: usize) -> f64 {
        self.samples_round_start as f64 / self.sample_rate as f64
            + (self.frame_offset + frame) as f64 * 0.03
    }

    fn sample_of(&self, frame: usize) -> u64 {
        self.samples_round_start + (self.frame_offset + frame) as u64 * self.frame_samples()
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn energy_dbfs(&self, start_sample: u64, end_sample: u64) -> Option<f64> {
        let lo = start_sample.saturating_sub(self.samples_round_start) as usize;
        let hi = end_sample.saturating_sub(self.samples_round_start) as usize;
        let (acc, n) = self.pcm_sum_sq(lo, hi);
        if n == 0 {
            return None;
        }
        dbfs_of_mean_square(acc / n as f64)
    }

    fn write_word(
        &self,
        out: &mut String,
        span: &Entry,
        with_conf: bool,
        with_evidence: bool,
        now: u64,
    ) {
        let (ss, es) = (
            self.sample_of(span.start_frame),
            self.sample_of(span.end_frame),
        );
        out.push('{');
        if with_conf {
            out.push_str("\"conf\": 1.0, ");
        }
        out.push_str(&format!(
            "\"end\": {}, ",
            escape_json_number(self.seconds(span.end_frame))
        ));
        out.push_str(&format!(
            "\"start\": {}, ",
            escape_json_number(self.seconds(span.start_frame))
        ));
        out.push_str("\"word\": ");
        write_string(
            out,
            match span.word {
                EntryWord::Word(w) => self.model.word(w),
                EntryWord::Silence => SIL,
                EntryWord::Speech => SPEECH,
            },
        );
        out.push_str(&format!(", \"start_sample\": {ss}, \"end_sample\": {es}"));
        if with_evidence {
            out.push_str(&format!(
                ", \"energy_dbfs\": {}",
                number_or_null(self.energy_dbfs(ss, es))
            ));
            let _ = now;
        }
        out.push('}');
    }

    fn push_floor(&self, out: &mut String) {
        if let Some(db) = self.floor.dbfs() {
            out.push_str(&format!(", \"floor_dbfs\": {}", escape_json_number(db)));
        }
    }

    /// A Mutex, not a cell, so the recognizer stays `Sync` for hosts that share it.
    fn spk_cache(&self) -> MutexGuard<'_, HashMap<Vec<u64>, Option<Arc<Pooled>>>> {
        self.spk_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Samples of the stream per speaker frame, and in one frame's window.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn spk_hop_and_window(&self) -> (usize, usize) {
        let o = &self.model.mfcc_opts;
        let at = |ms: f32| (f64::from(ms) * 0.001 * f64::from(self.sample_rate)).round() as usize;
        (at(o.frame_shift_ms), at(o.frame_length_ms))
    }

    /// The speaker evidence over spans of decoder frames of the current utterance, ascending and
    /// apart. With a floor margin set, a frame whose own energy is within it of the floor is
    /// not pooled. `[[rr:TD-14#Evidence is pooled over a span]]`
    fn spk_evidence(&self, spans: &[(usize, usize)]) -> Option<Arc<Pooled>> {
        let stream = self.spk_stream.as_ref()?;
        // The floor sorts its history on every read, so it is read only when a margin is set.
        let threshold = self
            .endpoint_floor_margin_db
            .and_then(|margin| self.floor.dbfs().map(|floor| floor + f64::from(margin)));
        let sf = self.model.conf.frame_subsampling_factor;
        let frame = |f: usize| ((self.frame_offset + f) * sf) as u64;
        let reach = spans.iter().map(|s| frame(s.1)).max().unwrap_or(0);
        // A partial looks up every entry's evidence on every block; the key of a few spans is
        // built where no allocation is needed to look it up.
        let mut short = [0u64; 2 + 2 * 4];
        let mut long: Vec<u64>;
        let n = 2 + 2 * spans.len();
        let key: &mut [u64] = if spans.len() <= 4 {
            &mut short[..n]
        } else {
            long = vec![0; n];
            &mut long[..]
        };
        key[0] = threshold.map_or(u64::MAX, f64::to_bits);
        key[1] = reach.min(stream.rows_end() as u64);
        for (i, &(a, b)) in spans.iter().enumerate() {
            key[2 + 2 * i] = frame(a);
            key[3 + 2 * i] = frame(b);
        }
        if let Some(hit) = self.spk_cache().get(&key[..]) {
            return hit.clone();
        }
        let frames = self.spk_frames(spans);
        let keep = |k: usize| threshold.is_none_or(|t| self.frame_dbfs(k).is_some_and(|e| e > t));
        let ev = stream.pool(&frames, keep).map(|ev| {
            let mut vector_json = String::new();
            Self::write_spk_vector_json(&mut vector_json, &ev.vector, ev.frames);
            let hop = self.spk_hop_and_window().0 as u64;
            let span_json = format!(
                "\"spk_start\": {}, \"spk_end\": {}",
                self.samples_round_start + ev.first as u64 * hop,
                self.samples_round_start + ev.end as u64 * hop
            );
            Arc::new(Pooled {
                vector_json,
                span_json,
            })
        });
        let mut cache = self.spk_cache();
        if cache.len() >= 1024 {
            cache.clear();
        }
        cache.insert(key.to_vec(), ev.clone());
        ev
    }

    /// The energy of the audio under stream frame `k`, over the frame's own window.
    fn frame_dbfs(&self, k: usize) -> Option<f64> {
        let (hop, window) = self.spk_hop_and_window();
        let (acc, n) = self.pcm_sum_sq(k * hop, k * hop + window);
        if n == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        dbfs_of_mean_square(acc / n as f64)
    }

    /// Speaker rows the decoder's next advance is likely to pool, computed on the block before it,
    /// supposing the next block is the size of this one, so the network does not run on the
    /// advance's block as well: the speech the best path or a reading is in at the frontier,
    /// from where it began, and the frames the advance will decode, all of them while a path is
    /// in speech, else those loud enough to be likely speech. Between advances, while the best
    /// path is in speech, its rows are kept up with a batch at a time, which leaves a final that
    /// flushes the stream fewer to compute. A row does not depend on when it is computed, so
    /// this moves compute and never a result. `[[rr:TD-14#Mean normalisation looks back only]]`
    fn spk_look_ahead(&mut self, next_block: usize) {
        let Some(s) = self.spk_stream.as_ref() else {
            return;
        };
        let (Some(dec), Some(pipe)) = (self.decoder.as_ref(), self.pipeline.as_ref()) else {
            return;
        };
        let sf = self.model.conf.frame_subsampling_factor;
        let decoded = dec.num_frames_decoded();
        let frontier = (self.frame_offset + decoded) * sf;
        let speech_start = |p: &Path| {
            self.entries(p)
                .last()
                .filter(|e| e.word != EntryWord::Silence && e.end_frame >= decoded)
                .map(|e| (self.frame_offset + e.start_frame) * sf)
        };
        let (hop, window) = self.spk_hop_and_window();
        let frames = |samples: usize| {
            samples
                .checked_sub(window)
                .map_or(0, |n| n / hop.max(1) + 1)
        };
        let chunk = self.model.conf.frames_per_chunk;
        let rctx = self.model.net.context.1;
        let chunks = |ready: usize| ready.saturating_sub(rctx) / chunk;
        let now = chunks(pipe.mfcc.num_frames_ready());
        let mut spans: Vec<(usize, usize)> = Vec::new();
        if chunks(frames(self.pcm_first + self.pcm.len() + next_block)) > now {
            // Without readings asked for, the look-ahead reads its own near the best, since the
            // best path takes up a word a close reading had first.
            let own;
            let readings: &[Path] = match self.readings.as_deref() {
                Some(r) if self.partial_alternatives > 0 => r,
                _ => {
                    let groups = dec.grouped(false);
                    let near = groups
                        .iter()
                        .take(LOOK_AHEAD_READINGS)
                        .take_while(|g| g.cost - groups[0].cost <= LOOK_AHEAD_LEAD)
                        .count();
                    own = dec.trace_groups(&groups, near);
                    &own
                }
            };
            let from = readings
                .iter()
                .filter_map(speech_start)
                .chain(self.spk_best_speech)
                .min();
            let boundary = (now + 1) * chunk;
            if let Some(f) = from.filter(|&f| f < frontier) {
                spans.push((f, frontier));
            }
            let threshold = self
                .floor
                .dbfs()
                .map_or(LOOK_AHEAD_DBFS, |floor| floor + LOOK_AHEAD_OVER_FLOOR_DB);
            for k in frontier..boundary {
                if from.is_some() || self.frame_dbfs(k).is_some_and(|e| e > threshold) {
                    let a = k.saturating_sub(LOOK_AHEAD_PAD).max(frontier);
                    let b = (k + 1 + LOOK_AHEAD_PAD).min(boundary);
                    match spans.last_mut() {
                        Some(last) if a <= last.1 => last.1 = b,
                        _ => spans.push((a, b)),
                    }
                }
            }
        } else {
            let in_speech = self.spk_best_speech.is_some();
            let end = s.rows_end() + s.pending_frames();
            if !in_speech || end < s.computed_to().max(frontier) + LOOK_AHEAD_BATCH {
                return;
            }
            spans.push((frontier, end));
        }
        if let Some(s) = self.spk_stream.as_mut() {
            s.catch_up();
            let _computed = s.compute(&spans);
            #[cfg(test)]
            tests::ROWS.with(|r| r.set((r.get().0 + _computed, r.get().1)));
        }
    }

    /// Bring the speaker features up to the audio before a result when it could pool a frame
    /// they have not reached, or when its spans could reach back to where the kept rows begin,
    /// which moves with the features.
    fn spk_catch_up_for_result(&mut self) {
        let sf = self.model.conf.frame_subsampling_factor;
        let decoded = self.decoder.as_ref().map_or(0, |d| d.num_frames_decoded());
        let (start, frontier) = (self.frame_offset * sf, (self.frame_offset + decoded) * sf);
        if let Some(s) = self.spk_stream.as_mut() {
            if frontier > s.rows_end()
                || s.rows_end() + s.pending_frames() >= start + crate::speaker::MAX_ROWS
            {
                s.catch_up();
            }
        }
    }

    /// Stream frames of spans of decoder frames of the current utterance.
    fn spk_frames(&self, spans: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let sf = self.model.conf.frame_subsampling_factor;
        spans
            .iter()
            .map(|&(a, b)| ((self.frame_offset + a) * sf, (self.frame_offset + b) * sf))
            .collect()
    }

    fn write_spk_vector(out: &mut String, p: &Pooled) {
        out.push_str(&p.vector_json);
    }

    fn write_spk_vector_json(out: &mut String, vector: &[f32], frames: usize) {
        out.push_str("\"spk\": [");
        for (i, v) in vector.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&escape_json_number(f64::from(*v)));
        }
        out.push_str(&format!("], \"spk_frames\": {frames}"));
    }

    /// The cache is cleared whenever the sample clock of `spk_start` restarts.
    fn write_spk_span(out: &mut String, p: &Pooled) {
        out.push_str(&p.span_json);
    }

    /// `, "spk": .., "spk_frames": .., "spk_start": .., "spk_end": ..` when the spans carry
    /// evidence, nothing otherwise.
    fn push_spk(&self, out: &mut String, spans: &[(usize, usize)]) {
        if let Some(ev) = self.spk_evidence(spans) {
            out.push_str(", ");
            Self::write_spk_vector(out, &ev);
            out.push_str(", ");
            Self::write_spk_span(out, &ev);
        }
    }

    /// The spans of a path's entries that are speech, words and `[speech]` alike.
    fn speech_spans(&self, path: &Path, words_only: bool) -> Vec<(usize, usize)> {
        self.entries(path)
            .iter()
            .filter(|e| match e.word {
                EntryWord::Word(_) => true,
                EntryWord::Speech => !words_only,
                EntryWord::Silence => false,
            })
            .map(|e| (e.start_frame, e.end_frame))
            .collect()
    }

    /// `push_spk` for one entry of a word list, or with TitaNet its span's evidence if embedded;
    /// `[sil]` carries none.
    fn push_entry_spk(&self, out: &mut String, e: &Entry) {
        if self.spk.is_none() || e.word == EntryWord::Silence {
            return;
        }
        let Some(sp) = self.spk_spans.as_ref() else {
            self.push_spk(out, &[(e.start_frame, e.end_frame)]);
            return;
        };
        let span = (self.sample_of(e.start_frame), self.sample_of(e.end_frame));
        if let Some(p) = sp.done.get(&span) {
            out.push_str(", ");
            Self::write_spk_vector(out, p);
            out.push_str(", ");
            Self::write_spk_span(out, p);
        }
    }

    /// With TitaNet set, queue the words the best path, and with reading jobs on the readings,
    /// show another entry after. A `[speech]` entry is always a path's last, so it is never
    /// queued; it carries evidence only where a queued word had its span.
    // [[rr:TD-15#The recognizer embeds a word once its span has closed]]
    fn spk_queue_closed_paths(&mut self) {
        if self.spk_spans.is_none() {
            return;
        }
        let mut closed: Vec<(usize, usize)> = Vec::new();
        let mut close = |entries: Vec<Entry>| {
            for pair in entries.windows(2) {
                if let EntryWord::Word(_) = pair[0].word {
                    closed.push((pair[0].start_frame, pair[0].end_frame));
                }
            }
        };
        if let Some(path) = self.best_path.as_ref() {
            close(self.entries(path));
        }
        if self.spk_reading_jobs {
            for path in self.readings.iter().flatten() {
                close(self.entries(path));
            }
        }
        for (a, b) in closed {
            self.spk_queue_span(a, b);
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn spk_queue_span(&mut self, a: usize, b: usize) {
        let span = (self.sample_of(a), self.sample_of(b));
        if self
            .spk_spans
            .as_ref()
            .is_none_or(|sp| sp.queued.contains(&span))
        {
            return;
        }
        let embed = if self.spk_trim_floor {
            self.spk_trimmed(span)
        } else {
            span
        };
        let at_16k = ((embed.1 - embed.0) as f64 * f64::from(crate::fbank::SAMPLE_RATE)
            / f64::from(self.sample_rate)) as usize;
        let frames = crate::fbank::num_frames(at_16k);
        let sp = self.spk_spans.as_mut().expect("checked above");
        sp.queued.insert(span);
        // At a faster rate the count is the resampler's, checked again when the job begins.
        if (crate::speaker::MIN_FRAMES - 1..=RECOGNIZER_MAX_FRAMES + 1).contains(&frames) {
            sp.queue.push_back(SpanJob { span, embed });
        }
    }

    /// A span less its frames within the floor margin of the floor at either edge, frames of
    /// 25 ms every 10 ms as TitaNet's.
    // [[rr:TD-15#The floor margin: decided by measurement]]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn spk_trimmed(&self, (mut lo, mut hi): (u64, u64)) -> (u64, u64) {
        let (Some(margin), Some(floor)) = (self.endpoint_floor_margin_db, self.floor.dbfs()) else {
            return (lo, hi);
        };
        let threshold = floor + f64::from(margin);
        let rate = f64::from(self.sample_rate);
        let (hop, window) = ((rate * 0.010).round() as u64, (rate * 0.025).round() as u64);
        let quiet = |a: u64, b: u64| self.energy_dbfs(a, b).is_none_or(|e| e <= threshold);
        while lo + window <= hi && quiet(lo, lo + window) {
            lo += hop;
        }
        while hi >= lo + window && quiet(hi - window, hi) {
            hi -= hop;
        }
        (lo, hi)
    }

    /// Run the queued embeddings for one budget of work. `[[rr:TD-15#Embedding runs in slices between advances]]`
    #[allow(clippy::cast_possible_truncation)]
    fn spk_slices(&mut self) {
        let Some(mut sp) = self.spk_spans.take() else {
            return;
        };
        let mut left = self.spk_slice_budget;
        while left > 0 {
            if sp.running.is_none() {
                let Some(j) = sp.queue.pop_front() else {
                    break;
                };
                let base = self.samples_round_start + self.pcm_first as u64;
                let (Some(lo), Some(hi)) =
                    (j.embed.0.checked_sub(base), j.embed.1.checked_sub(base))
                else {
                    continue;
                };
                if hi as usize > self.pcm.len() {
                    continue;
                }
                self.pcm
                    .normalized(lo as usize, hi as usize, &mut sp.samples);
                let begun = sp
                    .job
                    .begin_at_rate(sp.model, &sp.samples, self.sample_rate);
                let frames = sp.job.frames();
                if begun.is_err()
                    || !(crate::speaker::MIN_FRAMES..=RECOGNIZER_MAX_FRAMES).contains(&frames)
                {
                    continue;
                }
                sp.running = Some(j);
            }
            left = left.saturating_sub(sp.job.advance(sp.model, left));
            if sp.job.is_done() {
                let j = sp.running.take().expect("a job was running");
                let frames = sp.job.frames();
                let vector = sp.job.take();
                sp.done
                    .insert(j.span, self.titanet_pooled(j.embed.0, frames, &vector));
            }
        }
        self.spk_spans = Some(sp);
    }

    /// `[[rr:TD-15#What carries evidence]]`
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn titanet_pooled(&self, start: u64, frames: usize, vector: &[f32]) -> Pooled {
        let mut vector_json = String::new();
        Self::write_spk_vector_json(&mut vector_json, vector, frames);
        let reach = (frames - 1) * crate::fbank::FRAME_SHIFT + crate::fbank::FRAME_LENGTH;
        let reach = (reach as f64 * f64::from(self.sample_rate)
            / f64::from(crate::fbank::SAMPLE_RATE))
        .round() as u64;
        Pooled {
            vector_json,
            span_json: format!("\"spk_start\": {start}, \"spk_end\": {}", start + reach),
        }
    }

    /// Whether a phone belongs to a word rather than to silence, by `word_boundary.int`.
    fn is_word_phone(&self, phone: i32) -> bool {
        !matches!(
            self.model.word_boundary.get(&phone),
            None | Some(WordBoundary::Nonword)
        )
    }

    /// The text of a reading: its words, or `[speech]` for a wordless path that ends inside a
    /// word's phones, or `[sil]`.
    fn text_of(&self, words: &[Label], last_phone: Option<i32>) -> String {
        if words.is_empty() {
            return if last_phone.is_some_and(|p| self.is_word_phone(p)) {
                SPEECH.to_string()
            } else {
                SIL.to_string()
            };
        }
        words
            .iter()
            .map(|&w| self.model.word(w))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn text_of_path(&self, path: &Path) -> String {
        self.text_of(&path.words, path.phones.last().map(|s| s.phone))
    }

    /// The word list of a path: aligned words, and between or after them every run of silence
    /// phones as a `[sil]` entry. Leading silence before the first word is not an entry, except
    /// when the path has no word at all, where the whole run is. Word phones after the last
    /// aligned word are a `[speech]` entry: a word begun and not yet named.
    #[doc(hidden)]
    pub fn entries(&self, path: &Path) -> Vec<Entry> {
        let spans = self.align(path);
        let sil = &self.model.conf.silence_phones;
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for seg in &path.phones {
            if sil.contains(&seg.phone) {
                match runs.last_mut() {
                    Some(r) if r.1 == seg.start => r.1 = seg.end,
                    _ => runs.push((seg.start, seg.end)),
                }
            }
        }
        let mut out: Vec<Entry> = Vec::new();
        let first_word_start = spans.first().map(|s| s.start_frame);
        let mut ri = 0;
        for span in &spans {
            while ri < runs.len() && runs[ri].1 <= span.start_frame {
                if first_word_start.map(|f| runs[ri].0 >= f).unwrap_or(true) || !out.is_empty() {
                    out.push(Entry {
                        word: EntryWord::Silence,
                        start_frame: runs[ri].0,
                        end_frame: runs[ri].1,
                    });
                }
                ri += 1;
            }
            // a run overlapping the word is not silence between words
            while ri < runs.len() && runs[ri].0 < span.end_frame {
                ri += 1;
            }
            out.push(Entry {
                word: EntryWord::Word(span.word),
                start_frame: span.start_frame,
                end_frame: span.end_frame,
            });
        }
        while ri < runs.len() {
            out.push(Entry {
                word: EntryWord::Silence,
                start_frame: runs[ri].0,
                end_frame: runs[ri].1,
            });
            ri += 1;
        }
        let covered = spans.last().map(|s| s.end_frame).unwrap_or(0);
        let tail = path
            .phones
            .iter()
            .rev()
            .take_while(|seg| seg.start >= covered && self.is_word_phone(seg.phone))
            .fold(None, |acc: Option<(usize, usize)>, seg| {
                Some((seg.start, acc.map_or(seg.end, |a| a.1)))
            });
        if let Some((start, end)) = tail {
            out.push(Entry {
                word: EntryWord::Speech,
                start_frame: start,
                end_frame: end,
            });
        }
        out
    }

    /// libvosk's `PartialResult`, extended: the best path's words, the readings still alive when
    /// [`set_alternatives`](Self::set_alternatives) is on, word entries when
    /// [`set_partial_words`](Self::set_partial_words) is on, and the noise floor.
    /// Keys: <https://github.com/pyscape/utter/blob/main/docs/reference/results.md#the-partial-result>.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    pub fn partial(&mut self) -> &str {
        self.spk_catch_up_for_result();
        let empty = |this: &mut Self| {
            let mut out = format!("{{\"partial\": \"{SIL}\"");
            this.push_floor(&mut out);
            out.push('}');
            this.last_result = out;
        };
        if self.state != State::Running {
            empty(self);
            return &self.last_result;
        }
        let Some(dec) = self.decoder.as_ref() else {
            empty(self);
            return &self.last_result;
        };
        if dec.num_frames_decoded() == 0 {
            empty(self);
            return &self.last_result;
        }
        self.refresh_best_path();
        if self.partial_alternatives > 0 {
            self.refresh_readings();
        }
        let Some(path) = self.best_path.as_ref() else {
            empty(self);
            return &self.last_result;
        };
        let now = self.samples_round_start + self.samples_processed;
        let mut out = String::with_capacity(self.last_result.len());
        out.push_str("{\"partial\": ");
        write_string(&mut out, &self.text_of_path(path));
        if self.partial_alternatives > 0 {
            out.push_str(", \"partial_alternatives\": [");
            let empty: Vec<Path> = Vec::new();
            let alts = self.readings.as_ref().unwrap_or(&empty);
            let best: &[Label] = alts.first().map(|a| a.words.as_slice()).unwrap_or(&[]);
            for (i, alt) in alts.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str("{\"text\": ");
                write_string(&mut out, &self.text_of_path(alt));
                out.push_str(&format!(
                    ", \"confidence\": {}",
                    escape_json_number(-(alt.cost as f64))
                ));
                if self.partial_words {
                    out.push_str(", \"result\": [");
                    for (j, span) in self.entries(alt).iter().enumerate() {
                        if j > 0 {
                            out.push_str(", ");
                        }
                        self.write_word(&mut out, span, false, false, now);
                        out.pop();
                        self.push_entry_spk(&mut out, span);
                        out.push('}');
                    }
                    out.push(']');
                }
                out.push_str(&format!(
                    ", \"relation\": \"{}\", \"lead_delta\": {}",
                    relation(&alt.words, best),
                    number_or_null(
                        self.history
                            .get(&alt.words)
                            .and_then(|h| h.delta())
                            .map(f64::from)
                    ),
                ));
                out.push('}');
            }
            out.push(']');
        }
        let entries = if self.partial_words || self.spk.is_some() {
            self.entries(path)
        } else {
            Vec::new()
        };
        if self.partial_words {
            out.push_str(", \"partial_result\": [");
            for (j, span) in entries.iter().enumerate() {
                if j > 0 {
                    out.push_str(", ");
                }
                let mut w = String::new();
                self.write_word(&mut w, span, false, true, now);
                w.pop();
                let since = self.stable.get(j).map(|s| s.1).unwrap_or(now);
                let stable_ms =
                    ((now - since) as f64 / self.sample_rate as f64 * 1000.0).round() as u64;
                w.push_str(&format!(", \"stable_ms\": {stable_ms}"));
                out.push_str(&w);
                self.push_entry_spk(&mut out, span);
                out.push('}');
            }
            out.push(']');
        }
        self.push_floor(&mut out);
        if self.spk.is_some() {
            let speech: Vec<(usize, usize)> = entries
                .iter()
                .filter(|e| e.word != EntryWord::Silence)
                .map(|e| (e.start_frame, e.end_frame))
                .collect();
            self.push_spk(&mut out, &speech);
        }
        out.push('}');
        self.last_result = out;
        &self.last_result
    }

    fn final_json(&self, reason: Endpoint) -> String {
        let Some(dec) = self.decoder.as_ref() else {
            return "{\"text\": \"\"}".into();
        };
        if dec.num_frames_decoded() == 0 {
            return "{\"text\": \"\"}".into();
        }
        let now = self.samples_round_start + self.samples_processed;
        // A floor final is the path as it stood; a completed path would force a word onto it.
        let completed = reason != Endpoint::Floor;
        let mut out = String::from("{");
        if self.max_alternatives > 1 {
            out.push_str("\"alternatives\": [");
            let alts = dec.alternatives(completed, self.max_alternatives);
            for (i, alt) in alts.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!(
                    "{{\"confidence\": {}, ",
                    escape_json_number(-(alt.cost as f64))
                ));
                if self.words {
                    out.push_str("\"result\": [");
                    self.write_final_words(&mut out, alt, now);
                    out.push_str("], ");
                }
                out.push_str("\"text\": ");
                write_string(&mut out, &self.text_of_path(alt));
                out.push('}');
            }
            out.push(']');
            out.push_str(&format!(", \"endpoint\": \"{}\"", reason.label()));
            self.push_floor(&mut out);
            if let (Some(top), true) = (alts.first(), self.spk.is_some()) {
                self.push_spk(&mut out, &self.speech_spans(top, true));
            }
            out.push('}');
            return out;
        }
        let path = dec.best_path(completed).unwrap_or_default();
        if self.words {
            out.push_str("\"result\": [");
            self.write_final_words(&mut out, &path, now);
            out.push_str("], ");
        }
        // libvosk's keys where libvosk puts them, before the text; the span after the others.
        let ev = self
            .spk
            .and_then(|_| self.spk_evidence(&self.speech_spans(&path, true)));
        if let Some(ev) = ev.as_ref() {
            Self::write_spk_vector(&mut out, ev);
            out.push_str(", ");
        }
        out.push_str("\"text\": ");
        write_string(&mut out, &self.text_of_path(&path));
        out.push_str(&format!(", \"endpoint\": \"{}\"", reason.label()));
        self.push_floor(&mut out);
        if let Some(ev) = ev.as_ref() {
            out.push_str(", ");
            Self::write_spk_span(&mut out, ev);
        }
        out.push('}');
        out
    }

    /// libvosk's `Result`: the final of the utterance decoded so far; the next `accept`
    /// starts a new utterance. Keys: <https://github.com/pyscape/utter/blob/main/docs/reference/results.md#the-final-result>.
    pub fn result(&mut self) -> &str {
        self.spk_catch_up_for_result();
        if self.state != State::Running {
            self.last_result = "{\"text\": \"\"}".into();
            return &self.last_result;
        }
        self.update_stable();
        let decoded = self.decoder.as_ref().map(|d| d.num_frames_decoded());
        let reason = match self.last_endpoint.take() {
            Some((at, r)) if Some(at) == decoded => r,
            _ => Endpoint::Host,
        };
        self.state = State::Endpoint;
        self.last_result = self.final_json(reason);
        &self.last_result
    }

    /// libvosk's `FinalResult`: flush the pipeline, decode what remains, and drop the stream.
    /// Keys: <https://github.com/pyscape/utter/blob/main/docs/reference/results.md#the-final-result>.
    pub fn final_result(&mut self) -> &str {
        if self.state != State::Running {
            self.last_result = "{\"text\": \"\"}".into();
            return &self.last_result;
        }
        if let Some(p) = self.pipeline.as_mut() {
            p.input_finished();
        }
        if let Some(s) = self.spk_stream.as_mut() {
            s.finish();
        }
        self.update_silence_weights();
        self.advance_decoding();
        self.update_stable();
        self.last_endpoint = None;
        self.state = State::Finalized;
        self.last_result = self.final_json(Endpoint::Flush);
        // libvosk drops the pipeline here; the next accept rebuilds it.
        if let Some(d) = &self.decoder {
            self.frame_offset += d.num_frames_decoded();
            self.merges_carried += d.merges_close;
        }
        self.pipeline = None;
        self.nnet = None;
        self.decoder = None;
        self.spk_stream = None;
        &self.last_result
    }

    /// libvosk's `Reset`: end the utterance without producing a result.
    pub fn reset(&mut self) {
        self.state = State::Endpoint;
        self.last_result = "{\"text\": \"\"}".into();
    }

    /// Sample position, in audio fed since construction, of the last decoded frame's end.
    pub fn decoded_sample(&self) -> u64 {
        let decoded = self
            .decoder
            .as_ref()
            .map(|d| d.num_frames_decoded())
            .unwrap_or(0);
        self.samples_round_start + (self.frame_offset + decoded) as u64 * self.frame_samples()
    }

    /// Output frames decoded in the current utterance; 0 between utterances.
    pub fn num_frames_decoded(&self) -> usize {
        self.decoder
            .as_ref()
            .map(|d| d.num_frames_decoded())
            .unwrap_or(0)
    }

    /// Tokens alive in the beam at the last decoded frame; 0 between utterances.
    pub fn num_active_tokens(&self) -> usize {
        self.decoder.as_ref().map(|d| d.num_active()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    // The test signals are generated from sample counts and decibel levels.
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]

    use super::*;

    const RATE: f32 = 16000.0;

    #[test]
    fn a_recognizer_can_be_shared_across_threads() {
        fn shared<T: Send + Sync>() {}
        shared::<Recognizer<'static>>();
        shared::<SpeakerModel>();
    }

    fn amplitude(dbfs: f64) -> i16 {
        (32768.0 * 10f64.powf(dbfs / 20.0)).round() as i16
    }

    fn constant(dbfs: f64, seconds: f64) -> Vec<f32> {
        vec![f32::from(amplitude(dbfs)); (RATE as f64 * seconds) as usize]
    }

    /// Uniform over `[-peak, peak]`, from a fixed seed so the sequence repeats.
    fn noise(peak: i16, seconds: f64) -> Vec<i16> {
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        (0..(RATE as f64 * seconds) as usize)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (x >> 33) as f64 / (1u64 << 31) as f64;
                ((u * 2.0 - 1.0) * peak as f64).round() as i16
            })
            .collect()
    }

    fn raw(samples: &[i16]) -> Vec<f32> {
        samples.iter().map(|&s| f32::from(s)).collect()
    }

    #[test]
    fn the_floor_is_absent_until_a_window_exists() {
        let mut f = FloorTracker::new(RATE);
        f.feed(&vec![1000.0; 1599]);
        assert_eq!(f.dbfs(), None);
        f.feed(&[1000.0]);
        assert!(f.dbfs().is_some());
    }

    #[test]
    fn a_constant_signal_reads_its_own_level() {
        let mut f = FloorTracker::new(RATE);
        f.feed(&constant(-30.0, 1.0));
        let want = 20.0 * (amplitude(-30.0) as f64 / 32768.0).log10();
        assert!((f.dbfs().unwrap() - want).abs() < 0.01, "{:?}", f.dbfs());
    }

    #[test]
    fn a_floor_of_digital_silence_is_no_floor() {
        let mut f = FloorTracker::new(RATE);
        f.feed(&raw(&noise(180, 9.0)));
        f.feed(&constant(-999.0, 1.0));
        assert_eq!(f.dbfs(), None);
        f.feed(&raw(&noise(180, 10.0)));
        assert!((f.dbfs().unwrap() + 50.0).abs() < 1.0, "{:?}", f.dbfs());
    }

    #[test]
    fn a_quarter_second_of_digital_silence_does_not_take_the_floor() {
        let mut f = FloorTracker::new(RATE);
        // peak 180 is -50 dBFS RMS for a uniform sequence
        f.feed(&raw(&noise(180, 10.0)));
        f.feed(&vec![0.0; (RATE * 0.25) as usize]);
        assert!((f.dbfs().unwrap() + 50.0).abs() < 1.0, "{:?}", f.dbfs());
    }

    #[test]
    fn the_history_forgets_a_louder_room() {
        let mut f = FloorTracker::new(RATE);
        f.feed(&constant(-20.0, 10.0));
        f.feed(&constant(-60.0, 10.0));
        assert!((f.dbfs().unwrap() + 60.0).abs() < 1.0, "{:?}", f.dbfs());
    }

    #[test]
    fn history_stays_integer_until_a_sample_is_not_one() {
        let mut h = History::Int(Vec::new());
        h.extend(&[1.0, -0.0, -32768.0, 32767.0]);
        assert!(matches!(h, History::Int(_)));
        h.extend(&[32768.0]);
        assert!(matches!(&h, History::Float(v) if v[..] == [1.0, 0.0, -32768.0, 32767.0, 32768.0]));
        h.extend(&[0.5]);
        h.drop_front(1);
        assert_eq!(h.sum_sq(3, 5), (32768.0 * 32768.0 + 0.25, 2));
    }

    #[test]
    fn a_fraction_of_a_count_has_a_level() {
        let mut f = FloorTracker::new(RATE);
        f.feed(&vec![0.25; (RATE * 0.1) as usize]);
        let want = 20.0 * (0.25f64 / 32768.0).log10();
        assert!((f.dbfs().unwrap() - want).abs() < 1e-9, "{:?}", f.dbfs());
    }

    #[test]
    fn full_scale_and_everything_within_it_is_accepted() {
        let least = f32::from_bits(1);
        assert_eq!(check_normalized(&[]), Ok(()));
        assert_eq!(
            check_normalized(&[-1.0, -0.0, 0.0, least, -least, 0.5, 1.0]),
            Ok(())
        );
    }

    #[test]
    fn the_first_sample_refused_is_named_with_what_is_wrong_with_it() {
        let above = f32::from_bits(1.0f32.to_bits() + 1);
        for (bad, kind) in [
            (f32::NAN, AudioInputErrorKind::NotANumber),
            (f32::INFINITY, AudioInputErrorKind::Infinite),
            (f32::NEG_INFINITY, AudioInputErrorKind::Infinite),
            (above, AudioInputErrorKind::OutOfRange),
            (-above, AudioInputErrorKind::OutOfRange),
            (f32::MAX, AudioInputErrorKind::OutOfRange),
        ] {
            let mut block = vec![0.25f32; 16000];
            block[15999] = bad;
            assert_eq!(
                check_normalized(&block),
                Err(AudioInputError { index: 15999, kind })
            );
            block[3] = 2.0;
            assert_eq!(check_normalized(&block).unwrap_err().index, 3);
        }
        let e = AudioInputError {
            index: 7,
            kind: AudioInputErrorKind::OutOfRange,
        };
        assert_eq!(
            e.to_string(),
            "sample 7 of the block is outside [-1.0, 1.0]"
        );
    }

    fn group(words: &[Label], cost: f32, lead: Option<f32>) -> (Vec<Label>, f32, Option<f32>, i32) {
        (words.to_vec(), cost, lead, 0)
    }

    #[test]
    fn a_reading_is_born_carried_dropped_and_born_again() {
        let mut h = HashMap::new();
        let first = vec![group(&[1], 0.0, Some(2.0)), group(&[2], 2.0, Some(-2.0))];
        assert_eq!(roll_history(&mut h, &first).0, vec![None, None]);

        let second = vec![group(&[1], 0.0, Some(3.0)), group(&[2], 3.0, Some(-3.0))];
        assert_eq!(roll_history(&mut h, &second).0, vec![Some(1.0), Some(-1.0)]);

        let third = vec![group(&[1], 0.0, Some(1.0)), group(&[3], 1.0, Some(-1.0))];
        let (deltas, (lost, close)) = roll_history(&mut h, &third);
        assert_eq!(deltas, vec![Some(-2.0), None]);
        // [2] was three nats behind the leader when it went, so it is lost but not close
        assert_eq!((lost, close), (1, 0));
        assert_eq!(h.len(), 2);
        assert!(!h.contains_key(&vec![2]));

        let fourth = vec![group(&[1], 0.0, Some(1.0)), group(&[2], 1.0, Some(-1.0))];
        let (deltas, (lost, close)) = roll_history(&mut h, &fourth);
        assert_eq!(deltas, vec![Some(0.0), None]);
        // [3] was one nat behind when it went
        assert_eq!((lost, close), (1, 1));
    }

    #[test]
    fn one_surviving_reading_has_no_lead_and_no_delta() {
        let mut h = HashMap::new();
        assert_eq!(
            roll_history(&mut h, &[group(&[1], 0.0, Some(2.0))]).0,
            vec![None]
        );
        // the lead is undefined at this end, so the delta is null on both sides of it
        assert_eq!(
            roll_history(&mut h, &[group(&[1], 0.0, None)]).0,
            vec![None]
        );
        assert_eq!(
            roll_history(&mut h, &[group(&[1], 0.0, Some(2.0))]).0,
            vec![None]
        );
        assert_eq!(
            roll_history(&mut h, &[group(&[1], 0.0, Some(3.0))]).0,
            vec![Some(1.0)]
        );
    }

    #[test]
    fn a_relation_is_read_off_the_labels() {
        assert_eq!(relation(&[1, 2], &[1, 2]), "same");
        assert_eq!(relation(&[1], &[1, 2]), "prefix");
        assert_eq!(relation(&[], &[1, 2]), "prefix");
        assert_eq!(relation(&[1, 2, 3], &[1, 2]), "extends");
        assert_eq!(relation(&[1, 2], &[]), "extends");
        assert_eq!(relation(&[2, 3], &[1, 3]), "differs");
        assert_eq!(relation(&[], &[]), "same");
    }

    thread_local! {
        /// Speaker rows computed on this thread: ahead of the acoustic model's chunk, and when
        /// the best path's speech needed them.
        pub(super) static ROWS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
    }

    /// The stock small English model, named by `UTTER_TEST_MODEL`, and the speaker model.
    fn models() -> Option<(Model, SpeakerModel)> {
        let dir = std::path::PathBuf::from(std::env::var_os("UTTER_TEST_MODEL")?);
        Some((Model::open(&dir).ok()?, crate::speaker::tests::spk_model()?))
    }

    struct SpkRun<'a> {
        /// Block sizes in samples, taken in turn.
        blocks: &'a [usize],
        set_at: usize,
        alternatives: usize,
        margin: Option<f32>,
        partials: bool,
        /// Whether a final is read when the endpoint says so; a host may read on instead.
        finals: bool,
        eager: bool,
    }

    /// Every result a host reading after each block would see.
    fn spk_results(m: &Model, spk: &SpeakerModel, audio: &[i16], run: &SpkRun) -> Vec<String> {
        let grammar: Vec<String> = ["yes", "no", "seven", "stop", "go"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        crate::speaker::EAGER.with(|e| e.set(run.eager));
        let mut rec = Recognizer::new(m, RATE, &grammar).unwrap();
        rec.set_words(true);
        rec.set_partial_words(true);
        rec.set_alternatives(run.alternatives);
        rec.set_max_alternatives(run.alternatives);
        rec.set_endpoint_floor_margin(run.margin);
        let mut out = Vec::new();
        let mut set = false;
        let (mut at, mut i) = (0, 0);
        while at < audio.len() {
            let b = &audio[at..(at + run.blocks[i % run.blocks.len()]).min(audio.len())];
            if !set && at >= run.set_at {
                rec.set_spk_model(Some(spk)).unwrap();
                set = true;
            }
            if rec.accept(b).endpoint && run.finals {
                out.push(rec.result().to_string());
            } else if run.partials {
                out.push(rec.partial().to_string());
            }
            at += b.len();
            i += 1;
        }
        out.push(rec.final_result().to_string());
        crate::speaker::EAGER.with(|e| e.set(false));
        out
    }

    /// Eager and on-demand rows give the same results; the on-demand results.
    fn same_as_eager(m: &Model, spk: &SpeakerModel, audio: &[i16], mut run: SpkRun) -> Vec<String> {
        run.eager = true;
        let want = spk_results(m, spk, audio, &run);
        run.eager = false;
        let got = spk_results(m, spk, audio, &run);
        assert_eq!(want, got, "blocks {:?} set at {}", run.blocks, run.set_at);
        got
    }

    /// Two utterances apart by a second of quiet, so the network starts afresh past a gap.
    fn spk_audio() -> Vec<i16> {
        use crate::speaker::tests::clip;
        [clip("yes"), clip("seven"), noise(60, 1.0), clip("no")].concat()
    }

    /// 7, 30 and 100 ms blocks do not divide the acoustic model's 240 ms chunk, and a size that
    /// changes from block to block defeats the look-ahead's guess of the next.
    const BLOCKS: &[&[usize]] = &[
        &[160],
        &[640],
        &[3200],
        &[112],
        &[480],
        &[1600],
        &[480, 112, 1600, 640, 3200, 160, 2000],
    ];

    /// `[[rr:TD-14#Mean normalisation looks back only]]`: rows computed for the speech a result
    /// pools give every result byte for byte as rows computed for every frame as it arrives.
    #[test]
    fn rows_on_demand_give_every_result_as_rows_computed_eagerly() {
        let Some((m, spk)) = models() else { return };
        let audio = spk_audio();
        for (alternatives, margin) in [(0, None), (3, Some(6.0))] {
            for blocks in BLOCKS {
                for set_at in [0, 8000, 24000] {
                    let run = SpkRun {
                        blocks,
                        set_at,
                        alternatives,
                        margin,
                        partials: true,
                        finals: true,
                        eager: true,
                    };
                    let got = same_as_eager(&m, &spk, &audio, run);
                    assert!(got.iter().any(|r| r.contains("\"spk\": [")));
                }
            }
        }
    }

    #[test]
    fn a_final_read_alone_pools_every_row() {
        let Some((m, spk)) = models() else { return };
        let audio = spk_audio();
        for (alternatives, margin) in [(0, None), (3, Some(6.0))] {
            for blocks in [&BLOCKS[1], &BLOCKS[6]] {
                for set_at in [0, 24000] {
                    let run = SpkRun {
                        blocks,
                        set_at,
                        alternatives,
                        margin,
                        partials: false,
                        finals: true,
                        eager: true,
                    };
                    let got = same_as_eager(&m, &spk, &audio, run);
                    let words = got.iter().filter(|r| r.contains("\"word\": ")).count();
                    assert!(words >= 1, "{got:?}");
                    assert!(got
                        .iter()
                        .filter(|r| r.contains("\"word\": "))
                        .all(|r| r.contains("\"spk_frames\"")));
                }
            }
        }
    }

    /// Onsets the look-ahead sees coming and onsets it does not: loud words after digital
    /// silence, where there is no floor, and after a floor, and words too quiet for it after
    /// each; with readings asked for, and with the look-ahead reading its own.
    #[test]
    fn rows_ahead_of_the_chunk_give_every_result_as_rows_computed_eagerly() {
        let Some((m, spk)) = models() else { return };
        use crate::speaker::tests::clip;
        let quiet = |w: Vec<i16>| w.iter().map(|&v| v / 40).collect::<Vec<i16>>();
        let silence = |seconds: f64| vec![0i16; (RATE as f64 * seconds) as usize];
        let audio = [
            silence(0.6),
            clip("yes"),
            noise(60, 1.5),
            clip("seven"),
            noise(60, 1.0),
            quiet(clip("no")),
            silence(0.8),
            quiet(clip("yes")),
        ]
        .concat();
        let (mut ahead, mut needed) = (0, 0);
        for alternatives in [0, 3] {
            for blocks in BLOCKS {
                ROWS.with(|r| r.set((0, 0)));
                let run = SpkRun {
                    blocks,
                    set_at: 0,
                    alternatives,
                    margin: None,
                    partials: true,
                    finals: true,
                    eager: true,
                };
                same_as_eager(&m, &spk, &audio, run);
                let (a, n) = ROWS.with(std::cell::Cell::get);
                ahead += a;
                needed += n;
            }
        }
        assert!(
            ahead > 0 && needed > 0,
            "{ahead} ahead, {needed} when needed"
        );
    }

    /// A host that reads on past the endpoints keeps one utterance going beyond the 30 s of rows
    /// kept, so its evidence reaches back to where they begin.
    #[test]
    fn a_long_utterance_pools_only_the_kept_rows() {
        let Some((m, spk)) = models() else { return };
        use crate::speaker::tests::clip;
        let audio: Vec<i16> = (0..12)
            .flat_map(|_| [clip("yes"), clip("seven"), clip("no")].concat())
            .collect();
        // With a floor margin the evidence is pooled afresh as the floor moves, between the
        // acoustic model's chunks as well as on them.
        let run = SpkRun {
            blocks: &[640],
            set_at: 0,
            alternatives: 0,
            margin: Some(6.0),
            partials: true,
            finals: false,
            eager: true,
        };
        let got = same_as_eager(&m, &spk, &audio, run);
        let reach = |r: &String| {
            let (a, b) = (r.rfind("\"spk_start\": "), r.rfind("\"spk_end\": "));
            let num = |i: usize| -> u64 {
                r[i..]
                    .split([',', '}'])
                    .next()
                    .unwrap()
                    .split(": ")
                    .nth(1)
                    .unwrap()
                    .parse()
                    .unwrap()
            };
            a.zip(b).map(|(a, b)| num(b) - num(a))
        };
        // The evidence reaches the kept rows' 30 s and no further.
        let longest = got.iter().filter_map(reach).max().unwrap();
        assert!((29 * 16000..=30 * 16000).contains(&longest), "{longest}");
    }

    /// The stock small English model and TitaNet-small, named by `UTTER_TEST_MODEL` and
    /// `UTTER_TEST_TITANET_MODEL`.
    fn titanet_models() -> Option<(Model, SpeakerModel)> {
        let dir = std::path::PathBuf::from(std::env::var_os("UTTER_TEST_MODEL")?);
        let tn = std::path::PathBuf::from(std::env::var_os("UTTER_TEST_TITANET_MODEL")?);
        Some((Model::open(&dir).ok()?, SpeakerModel::open(&tn).ok()?))
    }

    struct TnRun<'a> {
        /// Block sizes in samples, taken in turn.
        blocks: &'a [usize],
        /// Where the model is set, in samples; `None` sets none.
        set_at: Option<usize>,
        readings: usize,
        margin: Option<f32>,
        reading_jobs: bool,
        trim: bool,
        /// A seed for a budget drawn afresh before every block, or the default budget.
        budgets: Option<u64>,
    }

    impl TnRun<'_> {
        fn plain(blocks: &[usize]) -> TnRun<'_> {
            TnRun {
                blocks,
                set_at: Some(0),
                readings: 0,
                margin: None,
                reading_jobs: false,
                trim: false,
                budgets: None,
            }
        }
    }

    /// Every result a host reading after each block would see, the final on each endpoint.
    fn tn_results(m: &Model, spk: &SpeakerModel, audio: &[i16], run: &TnRun) -> Vec<String> {
        let grammar: Vec<String> = ["yes", "no", "seven", "stop", "go"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut rec = Recognizer::new(m, RATE, &grammar).unwrap();
        rec.set_words(true);
        rec.set_partial_words(true);
        rec.set_alternatives(run.readings);
        rec.set_max_alternatives(run.readings);
        rec.set_endpoint_floor_margin(run.margin);
        rec.set_spk_reading_jobs(run.reading_jobs);
        rec.set_spk_trim_floor(run.trim);
        let mut seed = run.budgets.unwrap_or(0);
        let (mut out, mut at, mut i) = (Vec::new(), 0, 0);
        while at < audio.len() {
            let b = &audio[at..(at + run.blocks[i % run.blocks.len()]).min(audio.len())];
            if run.set_at == Some(at) || run.set_at.is_some_and(|s| s > at && s < at + b.len()) {
                rec.set_spk_model(Some(spk)).unwrap();
            }
            if run.budgets.is_some() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                rec.set_spk_slice_budget(1 + seed % (2 * SPK_SLICE_BUDGET));
            }
            if rec.accept(b).endpoint {
                out.push(rec.result().to_string());
            } else {
                out.push(rec.partial().to_string());
            }
            at += b.len();
            i += 1;
        }
        out.push(rec.final_result().to_string());
        out
    }

    /// A result less its speaker keys.
    fn without_spk(r: &str) -> String {
        let mut out = String::new();
        let mut rest = r;
        while let Some(i) = rest.find(", \"spk\": [") {
            out.push_str(&rest[..i]);
            let end = rest[i..].find("\"spk_end\": ").unwrap() + i + "\"spk_end\": ".len();
            let digits = rest[end..].find(|c: char| !c.is_ascii_digit()).unwrap();
            rest = &rest[end + digits..];
        }
        out.push_str(rest);
        out
    }

    fn key(r: &str, name: &str) -> u64 {
        let at = r.find(&format!("\"{name}\": ")).unwrap() + name.len() + 4;
        let n = r[at..]
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(r.len() - at);
        r[at..at + n].parse().unwrap()
    }

    /// Every entry's evidence is the stateless call's over the samples it names, written the
    /// same way; the spans that carry it.
    fn check_evidence(
        spk: &SpeakerModel,
        audio: &[i16],
        results: &[String],
    ) -> HashSet<(u64, u64)> {
        let mut want: HashMap<(u64, u64), String> = HashMap::new();
        let mut spans = HashSet::new();
        for r in results {
            let mut rest = r.as_str();
            while let Some(i) = rest.find("\"spk\": [") {
                let tail = &rest[i..];
                let (a, b) = (key(tail, "spk_start"), key(tail, "spk_end"));
                let got = &tail[..tail.find(", \"spk_start\"").unwrap()];
                let w = want.entry((a, b)).or_insert_with(|| {
                    let x: Vec<f32> = audio[a as usize..b as usize]
                        .iter()
                        .map(|&s| f32::from(s) / 32768.0)
                        .collect();
                    let v = spk.embed(&x).unwrap();
                    let json: Vec<String> = v
                        .iter()
                        .map(|&v| escape_json_number(f64::from(v)))
                        .collect();
                    format!(
                        "\"spk\": [{}], \"spk_frames\": {}",
                        json.join(", "),
                        crate::fbank::num_frames((b - a) as usize)
                    )
                });
                assert_eq!(got, w.as_str());
                spans.insert((a, b));
                rest = &tail[1..];
            }
        }
        spans
    }

    fn tn_audio() -> Vec<i16> {
        use crate::speaker::tests::clip;
        [
            clip("yes"),
            clip("seven"),
            noise(60, 1.0),
            clip("no"),
            clip("yes"),
            noise(60, 0.8),
            clip("seven"),
        ]
        .concat()
    }

    /// 10, 20, 40 and 200 ms, and sizes that change from block to block.
    const TN_BLOCKS: &[&[usize]] = &[&[160], &[320], &[640], &[3200], &[480, 112, 1600, 640]];

    /// `[[rr:TD-15#Embedding runs in slices between advances]]`
    #[test]
    // Miri perturbs float intrinsics by a few ULP, so the exact-bytes compare only holds natively.
    #[cfg_attr(miri, ignore)]
    fn titanet_evidence_is_the_stateless_embedding_in_any_blocks_and_slices() {
        let Some((m, spk)) = titanet_models() else {
            return;
        };
        let audio = tn_audio();
        let mut carried = 0;
        for blocks in TN_BLOCKS {
            for budgets in [None, Some(0x2545_f491), Some(7)] {
                for (readings, reading_jobs) in [(0, false), (3, false), (3, true)] {
                    let run = TnRun {
                        readings,
                        reading_jobs,
                        budgets,
                        ..TnRun::plain(blocks)
                    };
                    let got = tn_results(&m, &spk, &audio, &run);
                    carried += check_evidence(&spk, &audio, &got).len();
                    // No partial and no final carries evidence at its top level.
                    for r in &got {
                        let last = r.rfind('}').unwrap();
                        let top = r[..last].rfind(']').map_or(r.as_str(), |i| &r[i..]);
                        assert!(!top.contains("\"spk"), "{r}");
                    }
                }
            }
        }
        assert!(carried > 0);
    }

    /// `[[rr:TD-15#What carries evidence]]`
    #[test]
    #[cfg_attr(miri, ignore)]
    fn titanet_results_less_the_four_keys_are_the_results_without_a_model() {
        let Some((m, spk)) = titanet_models() else {
            return;
        };
        let audio = tn_audio();
        for blocks in [TN_BLOCKS[2], TN_BLOCKS[4]] {
            for (readings, margin, reading_jobs, trim) in [
                (0, None, false, false),
                (3, Some(8.0), true, false),
                (3, Some(8.0), false, true),
            ] {
                let run = TnRun {
                    readings,
                    margin,
                    reading_jobs,
                    trim,
                    ..TnRun::plain(blocks)
                };
                let with = tn_results(&m, &spk, &audio, &run);
                let without = tn_results(
                    &m,
                    &spk,
                    &audio,
                    &TnRun {
                        set_at: None,
                        ..run
                    },
                );
                assert!(with.iter().any(|r| r.contains("\"spk\": [")));
                let stripped: Vec<String> = with.iter().map(|r| without_spk(r)).collect();
                assert_eq!(stripped, without);
                check_evidence(&spk, &audio, &with);
            }
        }
    }

    /// Trimming cuts only floor-level frames at a span's edges, so what it embeds lies inside the
    /// span reported. `[[rr:TD-15#The floor margin: decided by measurement]]`
    #[test]
    #[cfg_attr(miri, ignore)]
    fn trimmed_evidence_lies_inside_its_span() {
        let Some((m, spk)) = titanet_models() else {
            return;
        };
        let audio = tn_audio();
        let run = TnRun {
            margin: Some(20.0),
            trim: true,
            ..TnRun::plain(TN_BLOCKS[2])
        };
        let got = tn_results(&m, &spk, &audio, &run);
        let mut inside = 0;
        for r in &got {
            for e in r.split("}, {").filter(|e| e.contains("\"spk_start\"")) {
                let (s, t) = (key(e, "start_sample"), key(e, "end_sample"));
                let (a, b) = (key(e, "spk_start"), key(e, "spk_end"));
                assert!(s <= a && b <= t, "{e}");
                inside += usize::from(s < a || b + 160 < t);
            }
        }
        assert!(inside > 0, "no span was trimmed");
        check_evidence(&spk, &audio, &got);
    }

    /// A model set mid-stream queues the words the partial has closed, so every span's evidence
    /// is the one a model set at the start gives.
    /// `[[rr:TD-15#The recognizer embeds a word once its span has closed]]`
    #[test]
    #[cfg_attr(miri, ignore)]
    fn titanet_set_mid_stream_gives_the_same_evidence() {
        let Some((m, spk)) = titanet_models() else {
            return;
        };
        let audio = tn_audio();
        let start = tn_results(&m, &spk, &audio, &TnRun::plain(TN_BLOCKS[2]));
        let from_start = check_evidence(&spk, &audio, &start);
        let without = tn_results(
            &m,
            &spk,
            &audio,
            &TnRun {
                set_at: None,
                ..TnRun::plain(TN_BLOCKS[2])
            },
        );
        for set_at in [8000, 24000] {
            let late = tn_results(
                &m,
                &spk,
                &audio,
                &TnRun {
                    set_at: Some(set_at),
                    ..TnRun::plain(TN_BLOCKS[2])
                },
            );
            let spans = check_evidence(&spk, &audio, &late);
            assert!(!spans.is_empty());
            assert!(spans.is_subset(&from_start));
            let stripped: Vec<String> = late.iter().map(|r| without_spk(r)).collect();
            assert_eq!(stripped, without);
        }
    }
}
