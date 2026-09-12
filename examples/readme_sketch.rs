//! The sketch in README.md, compiled. CI builds every target, so the README cannot drift
//! from the interface the way it had: it offered `rec.alternatives()` and
//! `step.end_of_speech`, neither of which the recognizer has ever had.
use std::path::Path;
struct Capture;
impl Capture {
    /// 40 ms of mono PCM, until the stream ends.
    fn next_block(&mut self) -> Option<&'static [i16]> {
        None
    }
}
fn main() -> std::io::Result<()> {
    let model = utter::Model::open(Path::new("vosk-model-small-en-us-0.15"))?;
    let grammar: Vec<String> = ["alpha", "bravo", "seven"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut rec = utter::Recognizer::new(&model, 16_000.0, &grammar)?;
    rec.set_words(true);
    rec.set_partial_words(true);
    rec.set_alternatives(4);

    let mut capture = Capture;
    while let Some(block) = capture.next_block() {
        let step = rec.accept(block);
        if step.endpoint {
            println!("{}", rec.result()); // the segment that just ended
        } else {
            println!("{}", rec.partial()); // best path, rivals, word spans
        }
        let _decoded_through = step.sample; // samples fed, as of the last decoded frame
    }
    println!("{}", rec.final_result());
    Ok(())
}
