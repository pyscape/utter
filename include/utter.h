/* utter: pure-Rust streaming decoder for Vosk models. C ABI mirroring libvosk's vosk_api.h.
 * Result strings stay valid until the next call on the same recognizer or its destruction. */
#ifndef UTTER_H
#define UTTER_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct UtterModel UtterModel;
typedef struct UtterSpkModel UtterSpkModel;
typedef struct UtterRecognizer UtterRecognizer;

/* Which runtime the host loaded: the crate version and the git revision it was built from
   (40 hex digits, "-dirty" appended for a build over uncommitted edits, "unknown" where no
   checkout or package record was there to ask). Both static; never freed. */
const char *utter_version(void);
const char *utter_revision(void);

UtterModel *utter_model_new(const char *path);
void utter_model_free(UtterModel *model);
int utter_model_find_word(const UtterModel *model, const char *word);

/* A speaker model directory as libvosk's vosk_spk_model_new reads it. A recognizer it is set on
   keeps it alive until the recognizer is freed or the model is replaced. */
UtterSpkModel *utter_spk_model_new(const char *path);
void utter_spk_model_free(UtterSpkModel *model);

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
/* A margin in dB over the reported floor within which the bound reads a wordless path as silence
   and a rival word at the floor cannot veto it; such a final names "floor" and carries no forced
   word. 0 or less removes the margin. */
void utter_recognizer_set_endpoint_floor_margin(UtterRecognizer *rec, float margin_db);
/* Speaker evidence on every result and word entry; NULL removes it. 0 on success, -1 on a null
   recognizer or a speaker model this audio cannot feed. */
int utter_recognizer_set_spk_model(UtterRecognizer *rec, const UtterSpkModel *spk);
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
