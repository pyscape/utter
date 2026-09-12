//! The C ABI, mirroring libvosk's `vosk_api.h` names so a host written against it maps one to
//! one. Strings returned by the result functions stay valid until the next call on the same
//! recognizer or its destruction, as libvosk's do.
// [[rr:TD-2#Interface]]

use crate::recognizer::RecognizerOptions;
use crate::{Model, Recognizer};
use std::ffi::{c_char, c_float, c_int, c_short, CStr, CString};
use std::path::Path;
use std::sync::Arc;

pub struct UtterModel {
    inner: Arc<Model>,
}

pub struct UtterRecognizer {
    // Declared first so it drops before the model handle it borrows from.
    inner: Recognizer<'static>,
    _model: Arc<Model>,
    last: CString,
}

fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

/// Open a model directory; null on failure.
#[no_mangle]
pub extern "C" fn utter_model_new(path: *const c_char) -> *mut UtterModel {
    let Some(path) = cstr(path) else { return std::ptr::null_mut() };
    match Model::open(Path::new(path)) {
        Ok(m) => Box::into_raw(Box::new(UtterModel { inner: Arc::new(m) })),
        Err(e) => {
            eprintln!("utter: {e}");
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn utter_model_free(model: *mut UtterModel) {
    if !model.is_null() {
        drop(unsafe { Box::from_raw(model) });
    }
}

/// Word id for a word, -1 when unknown.
#[no_mangle]
pub extern "C" fn utter_model_find_word(model: *const UtterModel, word: *const c_char) -> c_int {
    let (Some(m), Some(w)) = (unsafe { model.as_ref() }, cstr(word)) else { return -1 };
    m.inner.word_ids.get(w).map(|&i| i as c_int).unwrap_or(-1)
}

fn new_recognizer(model: *const UtterModel, sample_rate: c_float, grammar: *const c_char, unknown_cost: Option<f32>) -> *mut UtterRecognizer {
    let (Some(m), Some(g)) = (unsafe { model.as_ref() }, cstr(grammar)) else { return std::ptr::null_mut() };
    let words = match crate::json::parse_string_array(g) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("utter: {e}");
            return std::ptr::null_mut();
        }
    };
    let arc = m.inner.clone();
    // The Arc held next to the recognizer keeps the model alive for its lifetime.
    let model_ref: &'static Model = unsafe { &*Arc::as_ptr(&arc) };
    let opts = RecognizerOptions { unknown_cost, ..Default::default() };
    match Recognizer::with_options(model_ref, sample_rate, &words, &opts) {
        Ok(inner) => Box::into_raw(Box::new(UtterRecognizer { inner, _model: arc, last: CString::default() })),
        Err(e) => {
            eprintln!("utter: {e}");
            std::ptr::null_mut()
        }
    }
}

/// `grammar` is a JSON array of strings, as libvosk takes it. Null on failure.
#[no_mangle]
pub extern "C" fn utter_recognizer_new_grm(model: *const UtterModel, sample_rate: c_float, grammar: *const c_char) -> *mut UtterRecognizer {
    new_recognizer(model, sample_rate, grammar, None)
}

/// As `utter_recognizer_new_grm`, adding the model's unknown-word symbol with `unknown_cost`.
#[no_mangle]
pub extern "C" fn utter_recognizer_new_grm_unk(model: *const UtterModel, sample_rate: c_float, grammar: *const c_char, unknown_cost: c_float) -> *mut UtterRecognizer {
    new_recognizer(model, sample_rate, grammar, Some(unknown_cost))
}

#[no_mangle]
pub extern "C" fn utter_recognizer_free(rec: *mut UtterRecognizer) {
    if !rec.is_null() {
        drop(unsafe { Box::from_raw(rec) });
    }
}

#[no_mangle]
pub extern "C" fn utter_recognizer_set_words(rec: *mut UtterRecognizer, on: c_int) {
    if let Some(r) = unsafe { rec.as_mut() } {
        r.inner.set_words(on != 0);
    }
}

#[no_mangle]
pub extern "C" fn utter_recognizer_set_partial_words(rec: *mut UtterRecognizer, on: c_int) {
    if let Some(r) = unsafe { rec.as_mut() } {
        r.inner.set_partial_words(on != 0);
    }
}

/// Partial alternatives to report, 0 for none.
#[no_mangle]
pub extern "C" fn utter_recognizer_set_alternatives(rec: *mut UtterRecognizer, n: c_int) {
    if let Some(r) = unsafe { rec.as_mut() } {
        r.inner.set_alternatives(n.max(0) as usize);
    }
}

/// libvosk's `SetMaxAlternatives`: alternatives on finals, 0 keeps the plain shape.
#[no_mangle]
pub extern "C" fn utter_recognizer_set_max_alternatives(rec: *mut UtterRecognizer, n: c_int) {
    if let Some(r) = unsafe { rec.as_mut() } {
        r.inner.set_max_alternatives(n.max(0) as usize);
    }
}

/// 16-bit mono PCM, `length` samples. Returns 1 when an endpoint fired, 0 otherwise, -1 on a
/// null handle.
#[no_mangle]
pub extern "C" fn utter_recognizer_accept_waveform_s(rec: *mut UtterRecognizer, data: *const c_short, length: c_int) -> c_int {
    let Some(r) = (unsafe { rec.as_mut() }) else { return -1 };
    if data.is_null() || length <= 0 {
        return 0;
    }
    let samples = unsafe { std::slice::from_raw_parts(data, length as usize) };
    r.inner.accept(samples).endpoint as c_int
}

/// Sample position, in audio fed since construction, of the last decoded frame.
#[no_mangle]
pub extern "C" fn utter_recognizer_decoded_sample(rec: *const UtterRecognizer) -> u64 {
    unsafe { rec.as_ref() }.map(|r| r.inner.decoded_sample()).unwrap_or(0)
}

fn store(r: &mut UtterRecognizer, s: String) -> *const c_char {
    r.last = CString::new(s).unwrap_or_default();
    r.last.as_ptr()
}

#[no_mangle]
pub extern "C" fn utter_recognizer_partial_result(rec: *mut UtterRecognizer) -> *const c_char {
    let Some(r) = (unsafe { rec.as_mut() }) else { return std::ptr::null() };
    let s = r.inner.partial().to_string();
    store(r, s)
}

#[no_mangle]
pub extern "C" fn utter_recognizer_result(rec: *mut UtterRecognizer) -> *const c_char {
    let Some(r) = (unsafe { rec.as_mut() }) else { return std::ptr::null() };
    let s = r.inner.result().to_string();
    store(r, s)
}

#[no_mangle]
pub extern "C" fn utter_recognizer_final_result(rec: *mut UtterRecognizer) -> *const c_char {
    let Some(r) = (unsafe { rec.as_mut() }) else { return std::ptr::null() };
    let s = r.inner.final_result().to_string();
    store(r, s)
}

#[no_mangle]
pub extern "C" fn utter_recognizer_reset(rec: *mut UtterRecognizer) {
    if let Some(r) = unsafe { rec.as_mut() } {
        r.inner.reset();
    }
}
