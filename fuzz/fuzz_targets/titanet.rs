#![no_main]

use libfuzzer_sys::fuzz_target;
use utter::titanet::TitaNet;

// The seed corpus is a one-channel TitaNet with an embedding of one value.
const CONF: &str = "architecture=titanet\nsample_rate=16000\nfeature_normalize_type=per_feature\n\
window_size_ms=25\nwindow_stride_ms=10\nwindow_type=hann\nfeat_dim=80\ndim=1\n";

fuzz_target!(|data: &[u8]| {
    if let Ok(m) = TitaNet::from_bytes(CONF, data) {
        let x: Vec<f32> = (0..2_000u16)
            .map(|i| f32::from(i % 101) / 101.0 - 0.5)
            .collect();
        let _ = m.embed(&x);
    }
});
