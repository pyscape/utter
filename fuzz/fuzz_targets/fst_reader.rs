#![no_main]

use libfuzzer_sys::fuzz_target;
use utter::fst::{read_fst_bytes, SymbolTable};

fuzz_target!(|data: &[u8]| {
    if let Ok(f) = read_fst_bytes(data) {
        // Walking what was read is part of the surface: the state ids in the arcs are the file's.
        let mut fst = f.fst;
        fst.arc_sort_ilabel();
        fst.connect();
        for t in [f.isymbols, f.osymbols].into_iter().flatten() {
            let _ = t.id_to_symbol();
            let _ = t.symbol_to_id();
        }
    }
    let _ = SymbolTable::parse_text(&String::from_utf8_lossy(data)).id_to_symbol();
});
