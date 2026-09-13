# Reading a partial

The keys this page reads are listed in the
[results reference](../reference/results.md). The snippets are Python
on the JSON string `PartialResult` returns; the Rust and C bindings
return the same string, so the same reads apply to `partial` and
`utter_recognizer_partial_result` with whatever JSON parser you use.

Everything below is a line of Python on the partial you already have.
The thresholds are yours; the benchmark pages give the figures behind
each read and record the reads that did not survive measurement.

**Will this word hold?** On the Speech Commands test split, the first
word shown is later revised 12% of the time. The best predictor is the
lead of the top reading over the second. The scores are log-likelihoods
in nats, so the sigmoid of that gap is a usable probability that the
word survives:

```python
import json, math

def trust(partial_json):
    alts = json.loads(partial_json).get("partial_alternatives") or []
    if len(alts) < 2:
        return 1.0                                        # nothing else is close
    gap = alts[0]["confidence"] - alts[1]["confidence"]   # nats, >= 0
    return 1.0 / (1.0 + math.exp(-gap))

if trust(p) >= 0.9:                                       # a lead of about 2.2 nats
    act(json.loads(p)["partial"])
```

This is well calibrated in practice; the derivation and the measured
survival at each gap are in
[../benchmarks/partial-trust.md](../benchmarks/partial-trust.md).
Two cheaper ways to be surer: the entropy of `softmax(confidences)`
over the readings, which keeps ranking words correctly even at a fixed
gap; and simply waiting one more block, which costs 240 ms and by then
most revisions have already happened.

**Is a word coming?** A reading that `extends` the partial and is
gaining lead is the next word forming in the beam. The coming word is
often visible one advance before the partial grows, though its identity
is right only about one time in six at that point.

**Is the room quiet?** The `[sil]` reading leads and its `lead_delta`
hovers near zero. A leading `[speech]` reading on a quiet room is the
noise floor read as the start of a word; the floor, not the label,
settles it.

**Might the last word not be there?** The best `prefix` reading's lead
is the evidence against the tail. It is a competing sequence, not a
per-word probability; a competitor absent from the list is unknown, not
disproved.

**Has the speaker finished?** The span of the trailing `[sil]` entry,
against a threshold of your choosing. Or let the recognizer do it for
you, below.

**Was this final's word ever said?** On a quiet room the model can read
the noise floor as a word's first phone, and the 20 s length cap then
closes the stretch with a grammar word spanning it, as stock Vosk does.
That final says `rule5`, and its word carries a hold of zero because no
partial showed it. The same hold rule you apply to partial words applies
here.

```python
alts = json.loads(p)["partial_alternatives"]
top  = alts[0]

coming = [a for a in alts if a["relation"] == "extends" and (a["lead_delta"] or 0) > 0]
next_word = coming[0]["text"][len(top["text"]):].split()[:1] if coming else []

quiet = top["text"] == "[sil]" and abs(top["lead_delta"] or 0) < 0.5

prefix = next((a for a in alts if a["relation"] == "prefix"), None)
doubt  = prefix["confidence"] - top["confidence"] if prefix else None

sils  = [e for e in json.loads(p)["partial_result"] if e["word"] == "[sil]"]
tail  = sils[-1] if sils else None
ended = tail is not None and (tail["end_sample"] - tail["start_sample"]) / 16000 > 0.5

f = json.loads(final)
said = f["endpoint"] != "rule5" or any(w["stable_ms"] > 0 for w in f["result"])
```

What `lead_delta` does not do is improve the trust read: measured over
every reading the decoder tracks, it adds nothing to the gap. Use it for
the state reads above, not for trust. The figures are on the
[partial-states](../benchmarks/partial-states.md) page.

