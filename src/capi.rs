//! The C ABI, mirroring libvosk's `vosk_api.h` names so a host written against it maps one to
//! one. Strings returned by the result functions stay valid until the next call on the same
//! recognizer or its destruction, as libvosk's do. Every function that takes a handle is
//! `unsafe`: the caller guarantees the pointer came from this library and is still alive.
// [[rr:TD-2#Interface]]
#![allow(clippy::missing_safety_doc)]

use crate::recognizer::RecognizerOptions;
use crate::{Model, Recognizer};
use std::ffi::{c_char, c_float, c_int, c_short, CStr, CString};
use std::path::Path;
use std::sync::Arc;

/// A model handle; one per directory opened, shared by its recognizers.
pub struct UtterModel {
    inner: Arc<Model>,
}

/// A recognizer handle. It keeps the model alive and owns the last result string it returned.
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
    // SAFETY: non-null was checked; the entry points pass the host's NUL-terminated strings on.
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

/// Open a model directory; null on failure.
#[no_mangle]
pub unsafe extern "C" fn utter_model_new(path: *const c_char) -> *mut UtterModel {
    let Some(path) = cstr(path) else {
        return std::ptr::null_mut();
    };
    match Model::open(Path::new(path)) {
        Ok(m) => Box::into_raw(Box::new(UtterModel { inner: Arc::new(m) })),
        Err(e) => {
            eprintln!("utter: {e}");
            std::ptr::null_mut()
        }
    }
}

/// Free a model; null is ignored. Recognizers made from it keep it alive until they are freed.
#[no_mangle]
pub unsafe extern "C" fn utter_model_free(model: *mut UtterModel) {
    if !model.is_null() {
        // SAFETY: a non-null handle came from `utter_model_new` and is freed once.
        drop(unsafe { Box::from_raw(model) });
    }
}

/// Word id for a word, -1 when unknown.
#[no_mangle]
pub unsafe extern "C" fn utter_model_find_word(
    model: *const UtterModel,
    word: *const c_char,
) -> c_int {
    // SAFETY: null or a live handle from `utter_model_new`.
    let (Some(m), Some(w)) = (unsafe { model.as_ref() }, cstr(word)) else {
        return -1;
    };
    m.inner
        .word_ids
        .get(w)
        .and_then(|&i| c_int::try_from(i).ok())
        .unwrap_or(-1)
}

unsafe fn new_recognizer(
    model: *const UtterModel,
    sample_rate: c_float,
    grammar: *const c_char,
    unknown_cost: Option<f32>,
) -> *mut UtterRecognizer {
    // SAFETY: null or a live handle from `utter_model_new`.
    let (Some(m), Some(g)) = (unsafe { model.as_ref() }, cstr(grammar)) else {
        return std::ptr::null_mut();
    };
    let words = match crate::json::parse_string_array(g) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("utter: {e}");
            return std::ptr::null_mut();
        }
    };
    let arc = m.inner.clone();
    // SAFETY: `arc` is stored beside the recognizer and, by field order, outlives it, so the
    // model the reference points into is alive for every use of the recognizer.
    let model_ref: &'static Model = unsafe { &*Arc::as_ptr(&arc) };
    let opts = RecognizerOptions {
        unknown_cost,
        ..Default::default()
    };
    match Recognizer::with_options(model_ref, sample_rate, &words, &opts) {
        Ok(inner) => Box::into_raw(Box::new(UtterRecognizer {
            inner,
            _model: arc,
            last: CString::default(),
        })),
        Err(e) => {
            eprintln!("utter: {e}");
            std::ptr::null_mut()
        }
    }
}

/// `grammar` is a JSON array of strings, as libvosk takes it. Null on failure.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_new_grm(
    model: *const UtterModel,
    sample_rate: c_float,
    grammar: *const c_char,
) -> *mut UtterRecognizer {
    // SAFETY: the caller's guarantees on the handle and the string pass through unchanged.
    unsafe { new_recognizer(model, sample_rate, grammar, None) }
}

/// As `utter_recognizer_new_grm`, adding the model's unknown-word symbol with `unknown_cost`.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_new_grm_unk(
    model: *const UtterModel,
    sample_rate: c_float,
    grammar: *const c_char,
    unknown_cost: c_float,
) -> *mut UtterRecognizer {
    // SAFETY: the caller's guarantees on the handle and the string pass through unchanged.
    unsafe { new_recognizer(model, sample_rate, grammar, Some(unknown_cost)) }
}

/// Free a recognizer and the last result string it returned; null is ignored.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_free(rec: *mut UtterRecognizer) {
    if !rec.is_null() {
        // SAFETY: a non-null handle came from `new_recognizer` and is freed once.
        drop(unsafe { Box::from_raw(rec) });
    }
}

/// libvosk's `vosk_recognizer_set_words`: word entries on finals when `on` is nonzero.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_set_words(rec: *mut UtterRecognizer, on: c_int) {
    // SAFETY: null or a live handle from `new_recognizer`, used from one thread at a time.
    if let Some(r) = unsafe { rec.as_mut() } {
        r.inner.set_words(on != 0);
    }
}

/// libvosk's `vosk_recognizer_set_partial_words`: word entries on partials when `on` is nonzero.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_set_partial_words(rec: *mut UtterRecognizer, on: c_int) {
    // SAFETY: null or a live handle from `new_recognizer`, used from one thread at a time.
    if let Some(r) = unsafe { rec.as_mut() } {
        r.inner.set_partial_words(on != 0);
    }
}

/// A host endpoint bound: a final once the trailing silence reaches `trailing_ms` (0 or less
/// removes it), unless `extending_veto_nats` is positive and a reading extending the partial by a
/// further word is within that many nats of it.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_set_endpoint_bound(
    rec: *mut UtterRecognizer,
    trailing_ms: c_float,
    extending_veto_nats: c_float,
) {
    // SAFETY: null or a live handle from `new_recognizer`, used from one thread at a time.
    if let Some(r) = unsafe { rec.as_mut() } {
        r.inner.set_endpoint_bound(
            (trailing_ms > 0.0).then_some(trailing_ms),
            (extending_veto_nats > 0.0).then_some(extending_veto_nats),
        );
    }
}

/// Partial alternatives to report, 0 for none.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_set_alternatives(rec: *mut UtterRecognizer, n: c_int) {
    // SAFETY: null or a live handle from `new_recognizer`, used from one thread at a time.
    if let Some(r) = unsafe { rec.as_mut() } {
        r.inner.set_alternatives(usize::try_from(n).unwrap_or(0));
    }
}

/// libvosk's `SetMaxAlternatives`: alternatives on finals, 0 keeps the plain shape.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_set_max_alternatives(
    rec: *mut UtterRecognizer,
    n: c_int,
) {
    // SAFETY: null or a live handle from `new_recognizer`, used from one thread at a time.
    if let Some(r) = unsafe { rec.as_mut() } {
        r.inner
            .set_max_alternatives(usize::try_from(n).unwrap_or(0));
    }
}

/// 16-bit mono PCM, `length` samples. Returns 1 when an endpoint fired, 0 otherwise, -1 on a
/// null handle.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_accept_waveform_s(
    rec: *mut UtterRecognizer,
    data: *const c_short,
    length: c_int,
) -> c_int {
    // SAFETY: null or a live handle from `new_recognizer`, used from one thread at a time.
    let Some(r) = (unsafe { rec.as_mut() }) else {
        return -1;
    };
    let Ok(len) = usize::try_from(length) else {
        return 0;
    };
    if data.is_null() || len == 0 {
        return 0;
    }
    // SAFETY: `data` is non-null and, as libvosk's contract has it, points at `length` samples.
    let samples = unsafe { std::slice::from_raw_parts(data, len) };
    r.inner.accept(samples).endpoint as c_int
}

/// Sample position, in audio fed since construction, of the last decoded frame.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_decoded_sample(rec: *const UtterRecognizer) -> u64 {
    // SAFETY: null or a live handle from `new_recognizer`.
    unsafe { rec.as_ref() }
        .map(|r| r.inner.decoded_sample())
        .unwrap_or(0)
}

fn store(r: &mut UtterRecognizer, s: String) -> *const c_char {
    r.last = CString::new(s).unwrap_or_default();
    r.last.as_ptr()
}

/// libvosk's `vosk_recognizer_partial_result`: the partial as JSON, null on a null handle.
/// Keys: <https://github.com/pyscape/utter/blob/main/docs/reference/results.md#the-partial-result>.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_partial_result(
    rec: *mut UtterRecognizer,
) -> *const c_char {
    // SAFETY: null or a live handle from `new_recognizer`, used from one thread at a time.
    let Some(r) = (unsafe { rec.as_mut() }) else {
        return std::ptr::null();
    };
    let s = r.inner.partial().to_string();
    store(r, s)
}

/// libvosk's `vosk_recognizer_result`: the final of the utterance so far as JSON, null on a
/// null handle. Keys: <https://github.com/pyscape/utter/blob/main/docs/reference/results.md#the-final-result>.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_result(rec: *mut UtterRecognizer) -> *const c_char {
    // SAFETY: null or a live handle from `new_recognizer`, used from one thread at a time.
    let Some(r) = (unsafe { rec.as_mut() }) else {
        return std::ptr::null();
    };
    let s = r.inner.result().to_string();
    store(r, s)
}

/// libvosk's `vosk_recognizer_final_result`: flush the pipeline and return the final as JSON,
/// null on a null handle. Keys: <https://github.com/pyscape/utter/blob/main/docs/reference/results.md#the-final-result>.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_final_result(rec: *mut UtterRecognizer) -> *const c_char {
    // SAFETY: null or a live handle from `new_recognizer`, used from one thread at a time.
    let Some(r) = (unsafe { rec.as_mut() }) else {
        return std::ptr::null();
    };
    let s = r.inner.final_result().to_string();
    store(r, s)
}

/// libvosk's `vosk_recognizer_reset`: end the utterance without a result.
#[no_mangle]
pub unsafe extern "C" fn utter_recognizer_reset(rec: *mut UtterRecognizer) {
    // SAFETY: null or a live handle from `new_recognizer`, used from one thread at a time.
    if let Some(r) = unsafe { rec.as_mut() } {
        r.inner.reset();
    }
}
