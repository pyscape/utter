//! A Vosk model directory read once: the transition model and network, the `HCLr` graph with
//! its lookahead relabeling, the word symbols, the decoding configuration, and the i-vector
//! extractor when the model ships one.
// [[rr:TD-2#Inputs: the model directory]]
// [[rr:TD-2#Inputs: configuration]]

use crate::compose::{compose, erase_input_labels, ComposeError};
use crate::fst::{read_fst_file, read_fst_symbols, Label, SymbolTable, VectorFst};
use crate::grammar::grammar_fst;
use crate::ivector::IvectorInfo;
use crate::kaldi_io::err;
use crate::mfcc::MfccOptions;
use crate::nnet3::Nnet3;
use crate::transition_model::TransitionModel;
use std::collections::HashMap;
use std::io::Result;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct EndpointRule {
    pub must_contain_nonsilence: bool,
    pub min_trailing_silence: f32,
    pub max_relative_cost: f32,
    pub min_utterance_length: f32,
}

#[derive(Clone, Debug)]
pub struct ModelConf {
    pub min_active: usize,
    pub max_active: usize,
    pub beam: f32,
    pub lattice_beam: f32,
    pub acoustic_scale: f32,
    pub frame_subsampling_factor: usize,
    pub frames_per_chunk: usize,
    pub silence_phones: Vec<i32>,
    pub rules: [EndpointRule; 5],
}

impl Default for ModelConf {
    fn default() -> Self {
        let rule = |ns: bool, ts: f32, rc: f32, ul: f32| EndpointRule {
            must_contain_nonsilence: ns,
            min_trailing_silence: ts,
            max_relative_cost: rc,
            min_utterance_length: ul,
        };
        ModelConf {
            min_active: 200,
            max_active: 7000,
            beam: 16.0,
            lattice_beam: 10.0,
            acoustic_scale: 0.1,
            frame_subsampling_factor: 1,
            frames_per_chunk: 24,
            silence_phones: Vec::new(),
            rules: [
                rule(false, 5.0, f32::INFINITY, 0.0),
                rule(true, 0.5, 2.0, 0.0),
                rule(true, 1.0, 8.0, 0.0),
                rule(true, 2.0, f32::INFINITY, 0.0),
                rule(true, 0.0, f32::INFINITY, 20.0),
            ],
        }
    }
}

impl ModelConf {
    pub fn parse(txt: &str) -> ModelConf {
        let mut c = ModelConf::default();
        for line in txt.lines() {
            let line = line.trim();
            let Some(rest) = line.strip_prefix("--") else {
                continue;
            };
            let Some((k, v)) = rest.split_once('=') else {
                continue;
            };
            let v = v.trim();
            let f = || v.parse::<f32>().ok();
            let u = || v.parse::<usize>().ok();
            match k.trim() {
                "min-active" => c.min_active = u().unwrap_or(c.min_active),
                "max-active" => c.max_active = u().unwrap_or(c.max_active),
                "beam" => c.beam = f().unwrap_or(c.beam),
                "lattice-beam" => c.lattice_beam = f().unwrap_or(c.lattice_beam),
                "acoustic-scale" => c.acoustic_scale = f().unwrap_or(c.acoustic_scale),
                "frame-subsampling-factor" => {
                    c.frame_subsampling_factor = u().unwrap_or(c.frame_subsampling_factor)
                }
                "frames-per-chunk" => c.frames_per_chunk = u().unwrap_or(c.frames_per_chunk),
                "endpoint.silence-phones" => {
                    c.silence_phones = v.split(':').filter_map(|p| p.parse().ok()).collect();
                }
                key => {
                    if let Some(rest) = key.strip_prefix("endpoint.rule") {
                        let (n, field) = rest.split_once('.').unwrap_or(("", ""));
                        if let Ok(n) = n.parse::<usize>() {
                            if (1..=5).contains(&n) {
                                let r = &mut c.rules[n - 1];
                                match field {
                                    "must-contain-nonsilence" => {
                                        r.must_contain_nonsilence = v == "true"
                                    }
                                    "min-trailing-silence" => {
                                        r.min_trailing_silence =
                                            f().unwrap_or(r.min_trailing_silence)
                                    }
                                    "max-relative-cost" => {
                                        r.max_relative_cost = f().unwrap_or(r.max_relative_cost)
                                    }
                                    "min-utterance-length" => {
                                        r.min_utterance_length =
                                            f().unwrap_or(r.min_utterance_length)
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }
        // Kaldi rounds the chunk up to a multiple of the subsampling factor.
        while c.frames_per_chunk % c.frame_subsampling_factor != 0 {
            c.frames_per_chunk += 1;
        }
        c
    }
}

/// `word_boundary.int`: phone -> type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordBoundary {
    Begin,
    End,
    Internal,
    Singleton,
    Nonword,
}

/// How many compiled grammars a model keeps.
/// `[[rr:TD-4#The model keeps the grammars most recently asked of it]]`
pub const CACHED_GRAPHS: usize = 4;

#[derive(PartialEq, Eq, Hash)]
struct GraphKey {
    grammar: Vec<String>,
    unknown_cost: Option<u32>,
    max_states: usize,
}

pub struct Model {
    pub dir: PathBuf,
    pub conf: ModelConf,
    pub mfcc_opts: MfccOptions,
    pub tm: TransitionModel,
    pub net: Nnet3,
    pub hcl: VectorFst,
    pub relabel: HashMap<Label, Label>,
    pub reach: Vec<Vec<(Label, Label)>>,
    pub final_label: Label,
    pub disambig: Vec<Label>,
    pub words: Vec<String>,
    pub word_ids: HashMap<String, i64>,
    pub word_boundary: HashMap<i32, WordBoundary>,
    pub ivector: Option<IvectorInfo>,
    graphs: Mutex<Vec<(GraphKey, Arc<VectorFst>)>>,
}

impl Model {
    pub fn open(dir: &Path) -> Result<Model> {
        let conf = ModelConf::parse(&std::fs::read_to_string(dir.join("conf/model.conf"))?);
        let mfcc_opts =
            MfccOptions::from_conf(&std::fs::read_to_string(dir.join("conf/mfcc.conf"))?);
        let mdl = std::fs::read(dir.join("am/final.mdl"))?;
        let tm = TransitionModel::parse(&mdl)?;
        let net = Nnet3::parse(&mdl);
        let hclr = read_fst_file(&dir.join("graph/HCLr.fst"))?;
        let addon = hclr
            .addon
            .ok_or_else(|| err("HCLr.fst carries no lookahead relabeling"))?;
        let disambig: Vec<Label> = std::fs::read_to_string(dir.join("graph/disambig_tid.int"))?
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        let symbols = match read_fst_symbols(&dir.join("graph/Gr.fst")) {
            Ok((_, isyms, osyms)) => osyms.or(isyms),
            Err(_) => None,
        };
        let symbols = match symbols {
            Some(s) => s,
            None => SymbolTable::parse_text(&std::fs::read_to_string(dir.join("graph/words.txt"))?),
        };
        let words = symbols.id_to_symbol();
        let word_ids = symbols.symbol_to_id();
        let mut word_boundary = HashMap::new();
        if let Ok(txt) = std::fs::read_to_string(dir.join("graph/phones/word_boundary.int")) {
            for line in txt.lines() {
                let mut it = line.split_whitespace();
                if let (Some(p), Some(t)) = (it.next(), it.next()) {
                    let kind = match t {
                        "begin" => WordBoundary::Begin,
                        "end" => WordBoundary::End,
                        "internal" => WordBoundary::Internal,
                        "singleton" => WordBoundary::Singleton,
                        _ => WordBoundary::Nonword,
                    };
                    if let Ok(p) = p.parse() {
                        word_boundary.insert(p, kind);
                    }
                }
            }
        }
        let ivector = if dir.join("ivector/final.ie").exists() {
            Some(IvectorInfo::open(&dir.join("ivector"))?)
        } else {
            None
        };
        if net.ivector_dim > 0 && ivector.is_none() {
            return Err(err(
                "network wants an i-vector but the model has no extractor",
            ));
        }
        Ok(Model {
            dir: dir.to_path_buf(),
            conf,
            mfcc_opts,
            tm,
            net,
            hcl: hclr.fst,
            relabel: addon.label2index,
            reach: addon.intervals,
            final_label: addon.final_label,
            disambig,
            words,
            word_ids,
            word_boundary,
            ivector,
            graphs: Mutex::new(Vec::new()),
        })
    }

    /// The decoding graph for a grammar.
    /// `[[rr:TD-4#The model keeps the grammars most recently asked of it]]`
    pub fn grammar_graph(
        &self,
        grammar: &[String],
        unknown_cost: Option<f32>,
        max_states: usize,
        warn: impl FnMut(String),
    ) -> Result<Arc<VectorFst>> {
        let key = GraphKey {
            grammar: grammar.to_vec(),
            unknown_cost: unknown_cost.map(f32::to_bits),
            max_states,
        };
        // [[rr:TD-4#The lock spans the composition]]
        let mut cache = self.graphs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = cache.iter().position(|(k, _)| *k == key) {
            let hit = cache.remove(i);
            let graph = hit.1.clone();
            cache.push(hit);
            return Ok(graph);
        }
        let graph = Arc::new(self.compile_grammar(grammar, unknown_cost, max_states, warn)?);
        if cache.len() >= CACHED_GRAPHS {
            cache.remove(0);
        }
        cache.push((key, graph.clone()));
        Ok(graph)
    }

    /// Compile a grammar to the decoding graph: bigram, lookahead composition, erase the
    /// disambiguation labels. `unknown_cost` adds the model's unknown-word symbol to the grammar
    /// with that cost on its arcs. `warn` receives libvosk's warnings about unknown words.
    pub fn compile_grammar(
        &self,
        grammar: &[String],
        unknown_cost: Option<f32>,
        max_states: usize,
        warn: impl FnMut(String),
    ) -> Result<VectorFst> {
        let mut grammar: Vec<String> = grammar.to_vec();
        let unk = "[unk]";
        if unknown_cost.is_some() && !grammar.iter().any(|s| s.split(' ').any(|w| w == unk)) {
            grammar.push(unk.to_string());
        }
        let mut g = grammar_fst(&grammar, &self.word_ids, warn);
        if let (Some(cost), Some(&id)) = (unknown_cost, self.word_ids.get(unk)) {
            for st in g.states.iter_mut() {
                for a in st.arcs.iter_mut() {
                    if a.ilabel == id as i32 {
                        a.weight += cost;
                    }
                }
            }
        }
        if !self.relabel.is_empty() {
            for st in g.states.iter_mut() {
                for a in st.arcs.iter_mut() {
                    if a.ilabel != 0 {
                        a.ilabel = *self.relabel.get(&a.ilabel).unwrap_or(&-1);
                    }
                }
                st.arcs.sort_by_key(|a| a.ilabel);
            }
        }
        let mut fst = match compose(&self.hcl, &g, &self.reach, self.final_label, max_states) {
            Ok(f) => f,
            Err(ComposeError(m)) => return Err(err(&m)),
        };
        erase_input_labels(&mut fst, &self.disambig);
        Ok(fst)
    }

    /// The model's unknown-word symbol id, when the word table has one.
    pub fn unknown_word(&self) -> Option<Label> {
        self.word_ids.get("[unk]").map(|&i| i as Label)
    }

    pub fn word(&self, id: Label) -> &str {
        self.words
            .get(id as usize)
            .map(String::as_str)
            .unwrap_or("")
    }
}
