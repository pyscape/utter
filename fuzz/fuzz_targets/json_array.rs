#![no_main]

use libfuzzer_sys::fuzz_target;
use utter::json::{parse_string_array, write_string};

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    if let Ok(words) = parse_string_array(&text) {
        // What the parser accepts the writer must be able to write back.
        let mut out = String::new();
        for w in &words {
            write_string(&mut out, w);
        }
    }
});
