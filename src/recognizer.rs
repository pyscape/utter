//! The recognizer: libvosk's lifecycle over the streaming pipeline, the decoder, and the
//! result shapes libvosk emits, extended with sample-indexed intervals, per-word energy, hold
//! time, and partial alternatives.
// [[rr:TD-2#Interface]]
// [[rr:TD-2#The decoder: word intervals]]
// [[rr:TD-2#The decoder: per-word energy and silence]]
// [[rr:TD-2#The decoder: partial alternatives]]
// [[rr:TD-2#The decoder: finals]]

use crate::decoder::{Decoder, DecoderConfig, Path};
use crate::frontend::FeaturePipeline;
use crate::fst::{Label, VectorFst};
use crate::json::write_string;
use crate::looped::LoopedNnet;
use crate::model::{Model, WordBoundary};
use crate::silence_weighting::SilenceWeighting;
use std::collections::VecDeque;
use std::sync::Arc;

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

#[derive(Clone, Debug)]
pub struct WordSpan {
    pub word: Label,
    /// Output frames within the current utterance, `[start, end)`.
    pub start_frame: usize,
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

/// Construction options beyond libvosk's.
#[derive(Clone, Debug)]
pub struct RecognizerOptions {
    /// Add the model's unknown-word symbol to the grammar with this cost on its arcs.
    pub unknown_cost: Option<f32>,
    pub max_graph_states: usize,
    /// Weight for frames the decoder's best path calls silence; 1.0 turns it off.
    /// `[[rr:i-vector: no oracle on this machine]]`
    pub silence_weight: f32,
}

impl Default for RecognizerOptions {
    fn default() -> Self {
        RecognizerOptions {
            unknown_cost: None,
            max_graph_states: DEFAULT_MAX_GRAPH_STATES,
            silence_weight: SILENCE_WEIGHT,
        }
    }
}

/// One entry of a word list: a word span or a run of silence phones.
#[derive(Clone, Debug)]
pub struct Entry {
    /// Word id, or `None` for silence.
    pub word: Option<Label>,
    pub start_frame: usize,
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

    fn new(sample_rate: f32) -> Self {
        FloorTracker {
            hop: ((sample_rate as f64 * Self::HOP_SECONDS).round() as usize).max(1),
            hop_sum_sq: VecDeque::new(),
            open_sum_sq: 0.0,
            open_len: 0,
        }
    }

    fn feed(&mut self, samples: &[i16]) {
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

    fn dbfs(&self) -> Option<f64> {
        if self.hop_sum_sq.len() < 2 {
            return None;
        }
        let mut windows: Vec<f64> = self
            .hop_sum_sq
            .iter()
            .zip(self.hop_sum_sq.iter().skip(1))
            .map(|(a, b)| dbfs_of_mean_square((a + b) / (2 * self.hop) as f64))
            .collect();
        windows.sort_by(f64::total_cmp);
        Some(windows[windows.len() * 5 / 100])
    }
}

pub struct Recognizer<'m> {
    model: &'m Model,
    graph: Arc<VectorFst>,
    sample_rate: f32,
    state: State,
    pipeline: Option<FeaturePipeline<'m>>,
    nnet: Option<LoopedNnet<'m>>,
    decoder: Option<Decoder<'m>>,
    silence_weighting: SilenceWeighting,
    frame_offset: usize,
    samples_processed: u64,
    samples_round_start: u64,
    /// PCM fed since the pipeline began, for energy under words.
    pcm: Vec<i16>,
    floor: FloorTracker,
    words: bool,
    partial_words: bool,
    partial_alternatives: usize,
    max_alternatives: usize,
    /// Word at each partial position and the sample position since which it has held.
    stable: Vec<(Label, u64)>,
    /// The decoder's best path without final costs, the decoded frame count it was read at,
    /// and the counts the silence weighting's frame labels and the stable list were last
    /// brought up to. `[[rr:TD-7#Decision outcome]]`
    best_path: Option<Path>,
    best_path_frames: Option<usize>,
    sw_traceback_frames: Option<usize>,
    stable_frames: Option<usize>,
    last_result: String,
    log: Option<Box<dyn Fn(&str) + Send + Sync + 'm>>,
}

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

fn dbfs_of_mean_square(mean_square: f64) -> f64 {
    let rms = mean_square.sqrt();
    if rms <= 0.0 {
        -999.0
    } else {
        20.0 * (rms / 32768.0).log10()
    }
}

impl<'m> Recognizer<'m> {
    /// Compile the grammar and prepare a stream. An empty grammar is refused: decoding against
    /// the full language model is out of scope.
    pub fn new(model: &'m Model, sample_rate: f32, grammar: &[String]) -> std::io::Result<Self> {
        Self::with_options(model, sample_rate, grammar, &RecognizerOptions::default())
    }

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
            pcm: Vec::new(),
            floor: FloorTracker::new(sample_rate),
            words: false,
            partial_words: false,
            partial_alternatives: 0,
            max_alternatives: 0,
            stable: Vec::new(),
            best_path: None,
            best_path_frames: None,
            sw_traceback_frames: None,
            stable_frames: None,
            last_result: String::new(),
            log: None,
        })
    }

    pub fn graph(&self) -> &VectorFst {
        &self.graph
    }
    pub fn set_words(&mut self, on: bool) {
        self.words = on;
    }
    pub fn set_partial_words(&mut self, on: bool) {
        self.partial_words = on;
    }
    /// Partial alternatives to report, 0 for none.
    pub fn set_alternatives(&mut self, n: usize) {
        self.partial_alternatives = n;
    }
    /// Alternatives on finals, libvosk's `SetMaxAlternatives`; 0 keeps the plain shape.
    pub fn set_max_alternatives(&mut self, n: usize) {
        self.max_alternatives = n;
    }
    pub fn set_log_callback(&mut self, f: Box<dyn Fn(&str) + Send + Sync + 'm>) {
        self.log = Some(f);
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

    fn frame_samples(&self) -> u64 {
        // one output frame is subsampling x 10 ms
        (self.model.conf.frame_subsampling_factor as f64
            * self.model.mfcc_opts.frame_shift_ms as f64
            * 0.001
            * self.sample_rate as f64)
            .round() as u64
    }

    fn rebuild(&mut self) {
        self.samples_round_start += self.samples_processed;
        self.samples_processed = 0;
        self.frame_offset = 0;
        self.pcm.clear();
        self.forget_best_path();
        let mut opts = self.model.mfcc_opts.clone();
        opts.sample_rate = self.sample_rate;
        self.pipeline = Some(FeaturePipeline::new(&opts, self.model.ivector.as_ref()));
        self.nnet = Some(LoopedNnet::new(self.model));
        let cfg = self.decoder_config();
        self.decoder = Some(Decoder::new(
            self.graph.clone(),
            &self.model.tm.tid2pdf,
            &self.model.tm.tid2phone,
            cfg,
        ));
        self.stable.clear();
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
            }
            _ => self.rebuild(),
        }
    }

    fn forget_best_path(&mut self) {
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

    fn advance_decoding(&mut self) {
        let (Some(pipe), Some(nnet), Some(dec)) = (
            self.pipeline.as_mut(),
            self.nnet.as_mut(),
            self.decoder.as_mut(),
        ) else {
            return;
        };
        while dec.num_frames_decoded() < nnet.num_frames_ready(pipe) {
            let row = nnet.frame(pipe, dec.num_frames_decoded());
            dec.advance_frame(row);
        }
    }

    /// Feed 16-bit mono PCM at the recognizer's rate.
    pub fn accept(&mut self, samples: &[i16]) -> Step {
        if !(self.state == State::Running || self.state == State::Initialized) {
            self.clean_up();
        } else if self.pipeline.is_none() {
            self.rebuild();
        }
        self.state = State::Running;
        let step = (self.sample_rate * 0.2) as usize;
        let mut i = 0;
        while i < samples.len() {
            let end = (i + step).min(samples.len());
            let chunk: Vec<f32> = samples[i..end].iter().map(|&s| s as f32).collect();
            self.pipeline.as_mut().unwrap().accept(&chunk);
            self.update_silence_weights();
            self.advance_decoding();
            i = end;
        }
        self.pcm.extend_from_slice(samples);
        self.floor.feed(samples);
        self.samples_processed += samples.len() as u64;
        self.update_stable();
        let endpoint = self.endpoint_detected();
        let decoded = self
            .decoder
            .as_ref()
            .map(|d| d.num_frames_decoded())
            .unwrap_or(0);
        let sample =
            self.samples_round_start + (self.frame_offset + decoded) as u64 * self.frame_samples();
        Step { endpoint, sample }
    }

    /// libvosk's `EndpointDetected` over the current decoding.
    pub fn endpoint_detected(&self) -> bool {
        let Some(dec) = self.decoder.as_ref() else {
            return false;
        };
        let n = dec.num_frames_decoded();
        if n == 0 {
            return false;
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
        conf.rules.iter().any(|r| {
            (contains_nonsilence || !r.must_contain_nonsilence)
                && trailing >= r.min_trailing_silence
                && relative <= r.max_relative_cost
                && utterance >= r.min_utterance_length
        })
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
        let Some(tokens) = self.best_path.as_ref().map(|path| {
            self.entries(path)
                .iter()
                .map(|e| e.word.unwrap_or(-1))
                .collect::<Vec<Label>>()
        }) else {
            return;
        };
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

    /// Words only, libvosk's final word list.
    fn word_entries(&self, path: &Path) -> Vec<Entry> {
        self.align(path)
            .into_iter()
            .map(|s| Entry {
                word: Some(s.word),
                start_frame: s.start_frame,
                end_frame: s.end_frame,
            })
            .collect()
    }

    fn seconds(&self, frame: usize) -> f64 {
        self.samples_round_start as f64 / self.sample_rate as f64
            + (self.frame_offset + frame) as f64 * 0.03
    }

    fn sample_of(&self, frame: usize) -> u64 {
        self.samples_round_start + (self.frame_offset + frame) as u64 * self.frame_samples()
    }

    fn energy_dbfs(&self, start_sample: u64, end_sample: u64) -> f64 {
        let lo = start_sample.saturating_sub(self.samples_round_start) as usize;
        let hi = (end_sample.saturating_sub(self.samples_round_start) as usize).min(self.pcm.len());
        if hi <= lo {
            return -999.0;
        }
        let mut acc = 0.0f64;
        for &s in &self.pcm[lo..hi] {
            acc += (s as f64) * (s as f64);
        }
        dbfs_of_mean_square(acc / (hi - lo) as f64)
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
        write_string(out, span.word.map(|w| self.model.word(w)).unwrap_or(SIL));
        out.push_str(&format!(", \"start_sample\": {ss}, \"end_sample\": {es}"));
        if with_evidence {
            out.push_str(&format!(
                ", \"energy_dbfs\": {}",
                escape_json_number(self.energy_dbfs(ss, es))
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

    fn text_of(&self, words: &[Label]) -> String {
        if words.is_empty() {
            return SIL.to_string();
        }
        words
            .iter()
            .map(|&w| self.model.word(w))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The word list of a path: aligned words, and between or after them every run of silence
    /// phones as a `[sil]` entry. Leading silence before the first word is not an entry, except
    /// when the path has no word at all, where the whole run is.
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
                        word: None,
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
                word: Some(span.word),
                start_frame: span.start_frame,
                end_frame: span.end_frame,
            });
        }
        while ri < runs.len() {
            out.push(Entry {
                word: None,
                start_frame: runs[ri].0,
                end_frame: runs[ri].1,
            });
            ri += 1;
        }
        out
    }

    /// libvosk's `PartialResult`, extended.
    pub fn partial(&mut self) -> &str {
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
        let dec = self.decoder.as_ref().unwrap();
        let Some(path) = self.best_path.as_ref() else {
            empty(self);
            return &self.last_result;
        };
        let now = self.samples_round_start + self.samples_processed;
        let mut out = String::from("{\"partial\": ");
        write_string(&mut out, &self.text_of(&path.words));
        if self.partial_alternatives > 0 {
            out.push_str(", \"partial_alternatives\": [");
            let alts = dec.alternatives(false, self.partial_alternatives);
            for (i, alt) in alts.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str("{\"text\": ");
                write_string(&mut out, &self.text_of(&alt.words));
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
                    }
                    out.push(']');
                }
                out.push('}');
            }
            out.push(']');
        }
        if self.partial_words {
            out.push_str(", \"partial_result\": [");
            for (j, span) in self.entries(path).iter().enumerate() {
                if j > 0 {
                    out.push_str(", ");
                }
                let mut w = String::new();
                self.write_word(&mut w, span, false, true, now);
                w.pop();
                let since = self.stable.get(j).map(|s| s.1).unwrap_or(now);
                let stable_ms =
                    ((now - since) as f64 / self.sample_rate as f64 * 1000.0).round() as u64;
                w.push_str(&format!(", \"stable_ms\": {stable_ms}}}"));
                out.push_str(&w);
            }
            out.push(']');
        }
        self.push_floor(&mut out);
        out.push('}');
        self.last_result = out;
        &self.last_result
    }

    fn final_json(&self) -> String {
        let Some(dec) = self.decoder.as_ref() else {
            return "{\"text\": \"\"}".into();
        };
        if dec.num_frames_decoded() == 0 {
            return "{\"text\": \"\"}".into();
        }
        let now = self.samples_round_start + self.samples_processed;
        let mut out = String::from("{");
        if self.max_alternatives > 1 {
            out.push_str("\"alternatives\": [");
            let alts = dec.alternatives(true, self.max_alternatives);
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
                    for (j, span) in self.word_entries(alt).iter().enumerate() {
                        if j > 0 {
                            out.push_str(", ");
                        }
                        self.write_word(&mut out, span, false, false, now);
                    }
                    out.push_str("], ");
                }
                out.push_str("\"text\": ");
                write_string(&mut out, &self.text_of(&alt.words));
                out.push('}');
            }
            out.push(']');
            self.push_floor(&mut out);
            out.push('}');
            return out;
        }
        let path = dec.best_path(true).unwrap_or_default();
        if self.words {
            out.push_str("\"result\": [");
            for (j, span) in self.word_entries(&path).iter().enumerate() {
                if j > 0 {
                    out.push_str(", ");
                }
                // [[rr:TD-8#A final word carries its energy]]
                self.write_word(&mut out, span, true, true, now);
            }
            out.push_str("], ");
        }
        out.push_str("\"text\": ");
        write_string(&mut out, &self.text_of(&path.words));
        self.push_floor(&mut out);
        out.push('}');
        out
    }

    /// libvosk's `Result`: the final of the utterance decoded so far; the next `accept`
    /// starts a new utterance.
    pub fn result(&mut self) -> &str {
        if self.state != State::Running {
            self.last_result = "{\"text\": \"\"}".into();
            return &self.last_result;
        }
        self.state = State::Endpoint;
        self.last_result = self.final_json();
        &self.last_result
    }

    /// libvosk's `FinalResult`: flush the pipeline, decode what remains, and drop the stream.
    pub fn final_result(&mut self) -> &str {
        if self.state != State::Running {
            self.last_result = "{\"text\": \"\"}".into();
            return &self.last_result;
        }
        if let Some(p) = self.pipeline.as_mut() {
            p.input_finished();
        }
        self.update_silence_weights();
        self.advance_decoding();
        self.state = State::Finalized;
        self.last_result = self.final_json();
        // libvosk drops the pipeline here; the next accept rebuilds it.
        if let Some(d) = &self.decoder {
            self.frame_offset += d.num_frames_decoded();
        }
        self.pipeline = None;
        self.nnet = None;
        self.decoder = None;
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

    pub fn num_frames_decoded(&self) -> usize {
        self.decoder
            .as_ref()
            .map(|d| d.num_frames_decoded())
            .unwrap_or(0)
    }

    pub fn num_active_tokens(&self) -> usize {
        self.decoder.as_ref().map(|d| d.num_active()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f32 = 16000.0;

    fn amplitude(dbfs: f64) -> i16 {
        (32768.0 * 10f64.powf(dbfs / 20.0)).round() as i16
    }

    fn constant(dbfs: f64, seconds: f64) -> Vec<i16> {
        vec![amplitude(dbfs); (RATE as f64 * seconds) as usize]
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

    #[test]
    fn the_floor_is_absent_until_a_window_exists() {
        let mut f = FloorTracker::new(RATE);
        f.feed(&vec![1000i16; 1599]);
        assert_eq!(f.dbfs(), None);
        f.feed(&[1000]);
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
    fn a_quarter_second_of_digital_silence_does_not_take_the_floor() {
        let mut f = FloorTracker::new(RATE);
        // peak 180 is -50 dBFS RMS for a uniform sequence
        f.feed(&noise(180, 10.0));
        f.feed(&vec![0i16; (RATE * 0.25) as usize]);
        assert!((f.dbfs().unwrap() + 50.0).abs() < 1.0, "{:?}", f.dbfs());
    }

    #[test]
    fn the_history_forgets_a_louder_room() {
        let mut f = FloorTracker::new(RATE);
        f.feed(&constant(-20.0, 10.0));
        f.feed(&constant(-60.0, 10.0));
        assert!((f.dbfs().unwrap() + 60.0).abs() < 1.0, "{:?}", f.dbfs());
    }
}
