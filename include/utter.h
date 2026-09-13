/* utter: pure-Rust streaming decoder for Vosk models. C ABI mirroring libvosk's vosk_api.h.
 * Result strings stay valid until the next call on the same recognizer or its destruction. */
#ifndef UTTER_H
#define UTTER_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct UtterModel UtterModel;
typedef struct UtterRecognizer UtterRecognizer;

UtterModel *utter_model_new(const char *path);
void utter_model_free(UtterModel *model);
int utter_model_find_word(const UtterModel *model, const char *word);

/* grammar: a JSON array of strings, as libvosk takes it */
UtterRecognizer *utter_recognizer_new_grm(const UtterModel *model, float sample_rate, const char *grammar);
UtterRecognizer *utter_recognizer_new_grm_unk(const UtterModel *model, float sample_rate, const char *grammar, float unknown_cost);
void utter_recognizer_free(UtterRecognizer *rec);

void utter_recognizer_set_words(UtterRecognizer *rec, int on);
void utter_recognizer_set_partial_words(UtterRecognizer *rec, int on);
/* A host endpoint bound: a final once the trailing silence reaches trailing_ms (0 or less removes
   it), unless extending_veto_nats is positive and a reading extending the partial by a further word
   is within that many nats of it. */
void utter_recognizer_set_endpoint_bound(UtterRecognizer *rec, float trailing_ms, float extending_veto_nats);
void utter_recognizer_set_alternatives(UtterRecognizer *rec, int n);
void utter_recognizer_set_max_alternatives(UtterRecognizer *rec, int n);

/* 16-bit mono PCM; 1 when an endpoint fired, 0 otherwise, -1 on a null handle */
int utter_recognizer_accept_waveform_s(UtterRecognizer *rec, const short *data, int length);
uint64_t utter_recognizer_decoded_sample(const UtterRecognizer *rec);

const char *utter_recognizer_partial_result(UtterRecognizer *rec);
const char *utter_recognizer_result(UtterRecognizer *rec);
const char *utter_recognizer_final_result(UtterRecognizer *rec);
void utter_recognizer_reset(UtterRecognizer *rec);

#ifdef __cplusplus
}
#endif
#endif
