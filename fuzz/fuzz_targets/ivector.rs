#![no_main]

use libfuzzer_sys::fuzz_target;
use utter::ivector::{DiagGmm, IvectorExtractor};

// Both readers open on a fixed run of tokens the mutator is unlikely to build on its own; the
// leading byte prepends one so the bytes after it land in the body.
const PREFIXES: [&[u8]; 3] = [
    b"",
    b"\0B<DiagGMM> <WEIGHTS> ",
    b"\0B<IvectorExtractor> <w> ",
];

fuzz_target!(|data: &[u8]| {
    let (sel, body) = data.split_first().unwrap_or((&0, &[]));
    let mut bytes = PREFIXES[*sel as usize % PREFIXES.len()].to_vec();
    bytes.extend_from_slice(body);

    if let Ok(g) = DiagGmm::parse(&bytes) {
        let mut out = vec![0.0f32; g.num_gauss];
        let mut scratch = Vec::new();
        let x = vec![0.5f32; g.dim];
        g.loglikes(&x, &mut out, &mut scratch);
    }
    let _ = IvectorExtractor::parse(&bytes);
});
