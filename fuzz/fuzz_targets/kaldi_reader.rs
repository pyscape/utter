#![no_main]

use libfuzzer_sys::fuzz_target;
use utter::kaldi_io::{parse_text_matrix, KaldiReader};

// Every read the Kaldi binary reader offers, first in a fixed sequence over the bytes and then
// in the order the leading byte of the input picks, so a mutation reaches any one of them first.
fn drive(bytes: &[u8], order: u8) {
    let mut r = KaldiReader::new(bytes);
    let _ = r.expect_binary();
    let _ = r.read_token();
    let _ = r.read_i32();
    let _ = r.read_dim();
    let _ = r.read_f32();
    let _ = r.read_f64();
    let _ = r.read_bool();
    let _ = r.read_i32_vec();
    let _ = r.read_float_vec();
    let _ = r.read_float_matrix();
    let _ = r.read_double_vec();
    let _ = r.read_double_matrix();
    let _ = r.read_packed_double();

    let mut r = KaldiReader::new(bytes);
    for k in 0..8u8 {
        match order.rotate_left(u32::from(k)).wrapping_add(k) % 12 {
            0 => drop(r.read_token()),
            1 => drop(r.read_i32()),
            2 => drop(r.read_dim()),
            3 => drop(r.read_f32()),
            4 => drop(r.read_f64()),
            5 => drop(r.read_bool()),
            6 => drop(r.read_i32_vec()),
            7 => drop(r.read_float_vec()),
            8 => drop(r.read_float_matrix()),
            9 => drop(r.read_double_vec()),
            10 => drop(r.read_double_matrix()),
            _ => drop(r.read_packed_double()),
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let order = data.first().copied().unwrap_or(0);
    drive(data, order);
    let _ = parse_text_matrix(&String::from_utf8_lossy(data));
});
