//! Tests that need no model: file formats, the JSON writer, the FFT and GEMM kernels, and the
//! decoder over a hand-built graph.

use std::sync::Arc;
use utter::decoder::{Decoder, DecoderConfig};
use utter::fst::{read_fst_bytes, Arc as FstArc, SymbolTable, VectorFst};
use utter::ivector::IvectorExtractor;
use utter::kaldi_io::{parse_text_matrix, KaldiReader};
use utter::transition_model::TransitionModel;

fn arc(i: i32, o: i32, w: f32, n: u32) -> FstArc {
    FstArc {
        ilabel: i,
        olabel: o,
        weight: w,
        nextstate: n,
    }
}

#[test]
#[allow(clippy::cast_possible_truncation)]
fn wav_reader_reads_pcm16() {
    let dir = std::env::temp_dir().join(format!("utter-wav-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.wav");
    let samples: Vec<i16> = (0..100).map(|i| (i * 37 - 1800) as i16).collect();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36u32 + 200).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&16000u32.to_le_bytes());
    bytes.extend_from_slice(&32000u32.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&200u32.to_le_bytes());
    for s in &samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(&path, bytes).unwrap();
    let w = utter::wav::read_wav(&path).unwrap();
    assert_eq!(w.sample_rate, 16000);
    assert_eq!(w.channels, 1);
    assert_eq!(w.samples, samples);
}

#[test]
fn kaldi_binary_and_text_forms() {
    // \0B, token, FV of two floats, a size-prefixed i32, a bool
    let mut b = vec![0u8, b'B'];
    b.extend_from_slice(b"<Tok> FV ");
    b.push(4);
    b.extend_from_slice(&2i32.to_le_bytes());
    b.extend_from_slice(&1.5f32.to_le_bytes());
    b.extend_from_slice(&(-2.0f32).to_le_bytes());
    b.push(4);
    b.extend_from_slice(&7i32.to_le_bytes());
    b.push(b'T');
    let mut r = KaldiReader::new(&b[..]);
    r.expect_binary().unwrap();
    r.expect_token("<Tok>").unwrap();
    assert_eq!(r.read_float_vec().unwrap(), vec![1.5, -2.0]);
    assert_eq!(r.read_i32().unwrap(), 7);
    assert!(r.read_bool().unwrap());
    let (rows, cols, data) = parse_text_matrix(" [\n  1 2 3\n  4 5 6 ]").unwrap();
    assert_eq!((rows, cols), (2, 3));
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(parse_text_matrix("1 2 3").is_err());
}

#[test]
fn json_writer_escapes() {
    let mut s = String::new();
    utter::json::write_string(&mut s, "a\"b\\c\nd\u{1}");
    assert_eq!(s, "\"a\\\"b\\\\c\\nd\\u0001\"");
    let back = utter::json::parse_string_array(&format!("[{s}]")).unwrap();
    assert_eq!(back, vec!["a\"b\\c\nd\u{1}"]);
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn mfcc_frame_count_and_shape() {
    let opts = utter::mfcc::MfccOptions::from_conf(
        "--sample-frequency=16000\n--num-mel-bins=40\n--num-ceps=40\n--dither=0\n",
    );
    let mut m = utter::mfcc::Mfcc::new(&opts);
    assert_eq!(m.num_frames(399), 0);
    assert_eq!(m.num_frames(400), 1);
    assert_eq!(m.num_frames(16000), 98);
    let tone: Vec<f32> = (0..16000)
        .map(|i| (i as f32 * 0.1).sin() * 3000.0)
        .collect();
    let feats = m.compute(&tone);
    assert_eq!((feats.r, feats.c), (98, 40));
    assert!(feats.d.iter().all(|v| v.is_finite()));
    // online front end agrees with the batch computation frame for frame
    let mut online = utter::frontend::OnlineMfcc::new(&opts);
    for chunk in tone.chunks(640) {
        online.accept(chunk);
    }
    assert_eq!(online.frames.len(), 98);
    for (t, row) in online.frames.iter().enumerate() {
        for (a, b) in row.iter().zip(feats.row(t)) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}

/// Two words on a three-state graph with a silence self-loop; frames are chosen so the path
/// changes phone, emits words, and ends in silence.
fn word_graph() -> (Arc<VectorFst>, Vec<i32>, Vec<i32>) {
    // tid 1: silence phone 1 (pdf 0); tid 2: phone 2 word A (pdf 1); tid 3: phone 3 word B (pdf 2)
    let tid2pdf = vec![-1, 0, 1, 2];
    let tid2phone = vec![0, 1, 2, 3];
    let mut f = VectorFst::default();
    for _ in 0..3 {
        f.add_state();
    }
    f.start = 0;
    f.add_arc(0, arc(1, 0, 0.0, 0)); // silence loop
    f.add_arc(0, arc(2, 100, 1.0, 1)); // word A
    f.add_arc(1, arc(2, 0, 0.0, 1)); // A continues
    f.add_arc(1, arc(3, 200, 1.0, 2)); // word B
    f.add_arc(2, arc(3, 0, 0.0, 2));
    f.add_arc(2, arc(1, 0, 0.5, 2)); // trailing silence
    f.set_final(2, 0.0);
    (Arc::new(f), tid2pdf, tid2phone)
}

#[test]
fn decoder_traces_words_phones_and_trailing_silence() {
    let (fst, tid2pdf, tid2phone) = word_graph();
    let cfg = DecoderConfig {
        beam: 10.0,
        max_active: 100,
        min_active: 0,
        beam_delta: 0.5,
    };
    let mut dec = Decoder::new(fst, &tid2pdf, &tid2phone, cfg);
    let sil = [0.0, -5.0, -5.0];
    let a = [-5.0, 0.0, -5.0];
    let b = [-5.0, -5.0, 0.0];
    for frame in [sil, sil, a, a, a, b, b, sil, sil, sil] {
        dec.advance_frame(&frame);
    }
    let path = dec.best_path(false).unwrap();
    assert_eq!(path.words, vec![100, 200]);
    let phones: Vec<(i32, usize, usize)> = path
        .phones
        .iter()
        .map(|s| (s.phone, s.start, s.end))
        .collect();
    assert_eq!(phones, vec![(1, 0, 2), (2, 2, 5), (3, 5, 7), (1, 7, 10)]);
    assert_eq!(path.trailing_silence_frames(&[1]), 3);
    assert!(dec.final_relative_cost().is_finite());
    // alternatives: the best group is the partial's word sequence
    let alts = dec.alternatives(false, 5);
    assert_eq!(alts[0].words, vec![100, 200]);
    // a second run over the same frames is identical
    let (fst, tid2pdf, tid2phone) = word_graph();
    let mut dec2 = Decoder::new(
        fst,
        &tid2pdf,
        &tid2phone,
        DecoderConfig {
            beam: 10.0,
            max_active: 100,
            min_active: 0,
            beam_delta: 0.5,
        },
    );
    for frame in [sil, sil, a, a, a, b, b, sil, sil, sil] {
        dec2.advance_frame(&frame);
    }
    assert_eq!(dec2.best_path(false).unwrap().words, path.words);
}

#[test]
fn silence_weighting_deltas() {
    use utter::decoder::{Path, PhoneSegment};
    let mut sw = utter::silence_weighting::SilenceWeighting::new(&[1], 1e-3, 3);
    assert!(sw.active());
    // no traceback yet: new frames carry the silence weight
    let d = sw.get_delta_weights(6, 0);
    assert_eq!(d.len(), 6);
    assert!(d.iter().all(|&(_, w)| (w - 1e-3).abs() < 1e-9));
    // a traceback saying frame 0 is silence and frame 1 speech
    let path = Path {
        phones: vec![
            PhoneSegment {
                phone: 1,
                start: 0,
                end: 1,
            },
            PhoneSegment {
                phone: 2,
                start: 1,
                end: 2,
            },
        ],
        ..Default::default()
    };
    sw.compute_current_traceback(&path, 2);
    let d = sw.get_delta_weights(9, 0);
    // frames 3..6 (decoder frame 1) move from 1e-3 to 1.0
    let up: Vec<usize> = d
        .iter()
        .filter(|&&(_, w)| w > 0.5)
        .map(|&(f, _)| f)
        .collect();
    assert_eq!(up, vec![3, 4, 5, 6, 7, 8]);
}

// What the fuzz targets reached: a count or length the file states and the reader believed, an
// arithmetic step on a field at the end of its range, and an index the file chooses into a
// vector the reader built.

fn i32_field(v: i32) -> Vec<u8> {
    let mut b = vec![4u8];
    b.extend_from_slice(&v.to_le_bytes());
    b
}

fn after_binary(
    bytes: &[u8],
    f: impl Fn(&mut KaldiReader<&[u8]>) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let mut r = KaldiReader::new(bytes);
    r.expect_binary()?;
    f(&mut r)
}

#[test]
fn a_kaldi_length_buys_no_more_than_the_bytes_behind_it() {
    let mut fv = b"\0BFV ".to_vec();
    fv.extend_from_slice(&i32_field(i32::MAX));
    assert!(after_binary(&fv, |r| r.read_float_vec().map(|_| ())).is_err());

    let mut fm = b"\0BFM ".to_vec();
    fm.extend_from_slice(&i32_field(0x4000_0000));
    fm.extend_from_slice(&i32_field(0x4000_0000));
    assert!(after_binary(&fm, |r| r.read_float_matrix().map(|_| ())).is_err());

    let mut dp = b"\0BDP ".to_vec();
    dp.extend_from_slice(&i32_field(i32::MAX));
    assert!(after_binary(&dp, |r| r.read_packed_double().map(|_| ())).is_err());

    let mut iv = b"\0B".to_vec();
    iv.extend_from_slice(&i32_field(i32::MAX));
    assert!(after_binary(&iv, |r| r.read_i32_vec().map(|_| ())).is_err());
}

#[test]
fn a_transition_model_claims_no_more_tuples_than_it_holds() {
    let mut huge = b"<Tuples> ".to_vec();
    huge.extend_from_slice(&i32_field(i32::MAX));
    assert!(TransitionModel::parse(&huge).is_err());

    let mut extreme = b"<Tuples> ".to_vec();
    extreme.extend_from_slice(&i32_field(1));
    for field in [1, 0, i32::MAX, i32::MAX] {
        extreme.extend_from_slice(&i32_field(field));
    }
    let tm = TransitionModel::parse(&extreme).unwrap();
    assert_eq!(tm.num_tids(), 2);

    let mut negative = b"<Tuples> ".to_vec();
    negative.extend_from_slice(&i32_field(1));
    for field in [1, 0, -1, 0] {
        negative.extend_from_slice(&i32_field(field));
    }
    assert!(TransitionModel::parse(&negative).is_err());
}

#[test]
fn an_ivector_extractor_without_matrices_is_an_error() {
    let mut b = b"\0B<IvectorExtractor> <w> DM ".to_vec();
    b.extend_from_slice(&i32_field(0));
    b.extend_from_slice(&i32_field(0));
    b.extend_from_slice(b"<w_vec> DV ");
    b.extend_from_slice(&i32_field(0));
    b.extend_from_slice(b"<M> ");
    b.extend_from_slice(&i32_field(0));
    b.extend_from_slice(b"<SigmaInv> <IvectorOffset> ");
    b.push(8);
    b.extend_from_slice(&100.0f64.to_le_bytes());
    b.extend_from_slice(b"</IvectorExtractor> ");
    assert!(IvectorExtractor::parse(&b).is_err());
}

fn fst_header(fst_type: &[u8], flags: i32, start: i64, num_states: i64, num_arcs: i64) -> Vec<u8> {
    let mut b = 2_125_659_606i32.to_le_bytes().to_vec();
    for token in [fst_type, b"standard"] {
        b.extend_from_slice(&i32::try_from(token.len()).unwrap().to_le_bytes());
        b.extend_from_slice(token);
    }
    b.extend_from_slice(&2i32.to_le_bytes());
    b.extend_from_slice(&flags.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    for field in [start, num_states, num_arcs] {
        b.extend_from_slice(&field.to_le_bytes());
    }
    b
}

#[test]
fn an_fst_states_what_it_holds_and_points_only_inside_it() {
    assert!(read_fst_bytes(&fst_header(b"const", 0, 0, i64::MAX, 1)).is_err());
    assert!(read_fst_bytes(&fst_header(b"vector", 0, 0, i64::MAX, 1)).is_err());

    let mut arc_count = fst_header(b"const", 0, 0, 1, i64::MAX);
    arc_count.extend_from_slice(&[0u8; 20]);
    assert!(read_fst_bytes(&arc_count).is_err());

    // One state, one arc, and the arc names a state the file does not have.
    let mut past_the_end = fst_header(b"vector", 0, 0, 1, 1);
    past_the_end.extend_from_slice(&f32::INFINITY.to_le_bytes());
    past_the_end.extend_from_slice(&1i64.to_le_bytes());
    past_the_end.extend_from_slice(&1i32.to_le_bytes());
    past_the_end.extend_from_slice(&1i32.to_le_bytes());
    past_the_end.extend_from_slice(&0f32.to_le_bytes());
    past_the_end.extend_from_slice(&7u32.to_le_bytes());
    assert!(read_fst_bytes(&past_the_end).is_err());

    // "< 222222222": one line, and an id that would have sized a vector of 222 million names.
    let text = SymbolTable::parse_text("< 222222222\n");
    assert!(text.id_to_symbol().is_empty());
    assert_eq!(
        SymbolTable::parse_text("<eps> 0\none 1\n")
            .id_to_symbol()
            .len(),
        2
    );

    let mut symbols = fst_header(b"vector", 1, 0, 0, 0);
    symbols.extend_from_slice(&2_125_658_996i32.to_le_bytes());
    symbols.extend_from_slice(&0i32.to_le_bytes());
    symbols.extend_from_slice(&i64::MAX.to_le_bytes());
    symbols.extend_from_slice(&i64::MAX.to_le_bytes());
    assert!(read_fst_bytes(&symbols).is_err());
}
