#![no_main]

use libfuzzer_sys::fuzz_target;
use utter::transition_model::TransitionModel;

fuzz_target!(|data: &[u8]| {
    // The reader seeks the <Tuples> token, so give the mutator one to find.
    let mut bytes = Vec::with_capacity(data.len() + 9);
    if data.first().copied().unwrap_or(0) & 1 == 0 {
        bytes.extend_from_slice(b"<Tuples> ");
    }
    bytes.extend_from_slice(data);
    if let Ok(tm) = TransitionModel::parse(&bytes) {
        let n = tm.num_tids();
        for tid in 1..=n.min(1024) {
            let _ = tm.tid2pdf[tid];
            let _ = tm.tid2phone[tid];
        }
    }
});
