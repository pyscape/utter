# TD-9: Each reading carries its relation to the partial and the motion of its lead

- Tags: decoder, partials, alternatives, evidence, trust, measurement

## Context

What a host can read about a partial word's trust today is the gap
from the top reading's confidence to the next: its sigmoid predicts
whether the word will hold to within a few points across every bucket
of the public split, and nothing else on the partial ranks a first
sighting, `[[rr:Reading a partial's trust]]`. The gap is a snapshot: a
two-way softmax over the two leading beam readings' best-path costs,
`[[rr:The gap is a probability the API already gives you]]`, not a
lattice posterior, and the first word shown is still revised on 12.4%
of clips.

The first consumer's settle rule is a motion rule kept in the host: a
word is acted on after it has appeared unchanged in a fixed number of
consecutive partials, `[[rr:2. How the stock library is used]]`. The
use case asks for readiness where the evidence is,
`[[rr:3.4 What the snapshot was for]]`, and for that evidence from the
loop that decodes, `[[rr:5. What the application needs from this library]]`.

The question this record answers is what more a partial can carry
about how safe its words are, whether the four pieces
`[[rr:TD-2#Scope]]` keeps out, a lattice, its determinization, minimum
Bayes risk confidences and lattice n-best, are needed for it, and what
the runtime reports against what the host computes.

### What libvosk does with a lattice

Two partial paths. The plain one, `PartialResult` without partial
words, is `GetBestPath` without final costs: the best token's
traceback, words only. With partial words on, `PartialResult` takes a
lattice up to the frames the lattice has reached, word-aligns it and
runs `MinimumBayesRisk` for a per-word confidence and time, so a
partial-lattice path exists in the stock wheel; the use case measured
it trailing the audio by a median 0.965 s, which is what the live beam
answers. The final runs `GetLattice`, phone-pruned determinization,
then `MinimumBayesRisk` or `ShortestPath` for n alternatives.

### What a lattice at one frame can say

Every token in a frame-synchronous beam sits at the same frame, so
their forward costs compare, and a lattice over them has a well
defined posterior with the backward half set to zero at the frontier:
the standing of each reading given the audio fed so far, summed over
its alignments. That is the quantity the gap approximates with two
best paths, and the calibration says how closely. What no structure
over the tokens at one frame can say is whether a reading survives,
because that turns on audio not yet fed.

The half it lacks is observable one advance later. A posterior that
was exact would move unpredictably from one advance to the next; the
gap is not exact, so its errors persist across advances, and how it
moved is evidence about how it will move.

### Prior art

Every production stream commits a partial on hypothesis stability:
the partial traceback of Brown et al. (1982) commits the prefix all
live paths share, LocalAgreement commits the prefix the last k
hypotheses agree on, Google's `stability` and AWS's partial-result
stabilisation score a partial by how long it has stood. Each is a
hold, the figure `stable_ms` already carries. None reports how the
margin between the leading readings is moving.

## Considered options

- **A lattice for the partial: tokens kept per state, determinized,
  with minimum Bayes risk posteriors and n-best.** Deferred. Its
  posterior at a frame is the forward half the gap already
  approximates; the survival half is not in it; and no measured
  benefit for the fields below asks for it.
- **Summed costs beside the best-path cost on each token, a forward
  algorithm in the beam.** Rejected: at a merge the loser's mass joins
  the survivor's chain and is then attributed to the survivor's word
  sequence.
- **The host differences its partials.** Rejected: the host sees the n
  readings reported, so a reading's history before it entered them is
  invisible; five blocks in six change nothing and the host would have
  to find the advances itself; and every host writes the same tracker,
  the argument of `[[rr:TD-8#The runtime reports the floor]]`.
- **A survival probability from the runtime.** Rejected: the runtime
  reports evidence and never applies the bar,
  `[[rr:TD-8#The gate is the host's and is relative to the floor]]`;
  the calibration lives on the benchmark page.
- **A silence state or a next-word field on the partial.** Rejected:
  each is a bar on a lead, a motion and a span the partial already
  carries.
- **A count of lead changes.** Rejected: a lead change moves the
  partial's words, which resets `stable_ms` at the position that
  moved.
- **A convergence figure over the readings, the entropy of their
  softmax.** Not a field: a function of the confidences one partial
  already carries. It is the strongest second read a host has, 0.88
  rank-AUC against revision beside the gap's 0.88 and 0.59 to 0.69
  within every bucket of the gap; the README gives it beside the
  sigmoid.
- **An age on a reading.** Rejected: how long a reading has stood is
  worth nothing beyond the hold and the motion (0.54 to 0.56 within the
  gap's buckets, against the gap's own 0.88), and the benchmark carries
  no age for it.
- **A rate in nats per second.** Rejected: the readings change only at
  an advance, so the advance is the series' clock and a delta per
  advance is what a rate would be computed from.

## Decision outcome

### No lattice is added for this feature

The partial's readings remain the beam grouping of
`[[rr:TD-2#The decoder: partial alternatives]]`. No measured benefit
justifies adding a lattice, determinization or posteriors for the
fields below, and the question is deferred, not closed: the
disappearance proxy the page reports (a reading within two nats of the
leader absent at the next advance, before 56% of first sightings,
revised on 7.7% against 18.6%) observes the readings reported, not the
histories a merge destroys or the alignment mass a best path drops,
and its correlation with survival does not show those losses harmless.
Retained histories or posterior mass are evaluated, if at all, by a
measurement of their own.

### Readings are read once per decoding advance

An advance is one chunk of the network decoded,
`[[rr:TD-2#The network]]`: 240 ms of audio at the stock small models'
chunk, the first at 400 ms. With alternatives requested, the recognizer
groups the surviving tokens by word sequence once per advance, inside
the drain that decodes the chunk, so an `accept` that decodes several
chunks samples each. The history is kept over every group, not only
the n reported: the output limit never restricts it, so a reading
entering the top n carries the motion it earned outside it, and one
alternative requested still tracks every group that survives. The
groups and the traced top n are held with the decoded frame count as
the best path is, `[[rr:TD-7#Decision outcome]]`, and `partial` reads
them from there. One grouping serves both the history and the trace.
A change of n invalidates the trace, not the history; restarting the
decoder clears the history; the lead is undefined only when one group
survives.

### Every reading carries its lead's motion

A reading's identity is its word sequence, the grouping key; the empty
sequence is the reading `[sil]`. Its lead is its confidence less the
best confidence among the other readings: the gap for rank 0, its
deficit for every other, and undefined when there is one reading. Each
reading carries `lead_delta`, its lead at this advance less its lead at
the previous one, in nats, positive when it gained on the field and
negative when it lost; null when the reading was absent at the previous
advance or the lead was undefined at either end. A sequence that leaves
the groups and returns is new again.

### Every reading names its relation to the partial

Each reading carries `relation`, its word sequence set against rank
0's: `same` on rank 0; `prefix` when its words are a proper prefix of
the partial's, the empty reading `[sil]` included; `extends` when the
partial's words are a proper prefix of its; `differs` otherwise. These
are sequence relations, exact from the labels, and nothing more: a
`prefix` reading is a competing sequence without the partial's tail,
not a posterior for a null at an aligned word position, and `differs`
says nothing about which word differs (`alpha seven` against `bravo
seven` is `differs` though both end in `seven`). The confusion network
of `[[rr:3.2 Null evidence: the confusion network with the empty hypothesis]]`
is not what this is. The README shows a host what each is for: a
`prefix` reading competes by omitting the partial's tail, a `differs`
reading offers another phrase, an `extends` reading offers a possible
continuation; a null `lead_delta` is a broken history, an absent
competitor is unknown evidence, not evidence of absence; and the
trailing silence entry of `[[rr:TD-8#Decision outcome]]` remains the
clock beside them. Per-word readiness, the consumer's objective, stays
visible as work to validate on its own recordings, not a claim of
these keys. The bar in any read stays the host's.

### The runtime reports what needs history it alone holds

A figure that needs the previous advance, or a group outside the n
reported, is the runtime's, and so is a label the runtime holds exactly
where a host would rebuild it from text; a figure that is a function of
the numbers one partial already carries is the host's. The gap, its
sigmoid and the readings' entropy are the host's. The runtime never
applies a bar to any of them.

### The keys are appended

`relation` and `lead_delta` follow the keys of `[[rr:TD-2#Interface]]`
at the end of each entry of `partial_alternatives`, in that order,
after `result` where partial words are on, so the order that record
fixes is a prefix of what is emitted. With the two keys removed the
output is byte-identical to today's. Finals carry no series.

### The benchmark is paired, held out, and charged in milliseconds

The page `[[rr:Reading a partial's trust]]` is the before and the
after. The target is fixed: whether the first word shown survives to
the final, on the testing split, the same clips for every rule. Before
is the gap's sigmoid as the README gives it and a gap recalibrated on
validation, with entropy and a hold-only rule as further baselines;
after adds the motion. Every coefficient and threshold, the operating
points included, is fitted on the validation split and scored on the
testing split, never chosen on the clips it is scored on. Each rule is
scored three ways: rank-AUC against revision, with a paired bootstrap
interval on the difference between rules; calibration in the gap's
buckets; and the host's trade, revisions caught against the delay in
milliseconds of audio from the sighting to the block the rule first
clears, compared at equal delay: operating targets are chosen on
validation, the realized testing delay and catch are reported with
intervals, and an exact McNemar test is taken at the README's
operating point. Every rule is scored at the first sighting and again
one advance later, and the page reports how often each figure is
defined at each moment, on the runtime's all-group history as well as
the harness's reported readings, since motion is often null and its
availability is part of its worth. Whether a motion figure earns its key
is decided on the two pages together: on this one by separating
survivors from revisions within a bucket of the gap, and on the states
page by the reads it supports there. A
later change that lowers revisions caught at equal delay is a
finding, so that waiting longer can never pass as better evidence. The runtime's fields are checked
against a diagnostic trace that observes every group at an explicit
decoding position, within a stated tolerance, not against the harness's
top-n series. The states page measures candidate preview apart from
onset prediction: whether the coming word's reading is in the beam
before rank 0 grows, how long before, and how often its identity is
right, with candidate presence and lead compared against motion. Every
figure is kept in machine form beside the page.

## Consequences

- Each reading entry gains two keys. Partial text, segments and word
  times are untouched, so byte identity over the gate corpus holds
  without alternatives, and holds with them once the two keys are
  removed. The grouping's cost is measured with alternatives on and
  off.
- At a first sighting the leading reading is new on 59.3% of clips and
  its own motion null among the readings reported; the runtime's history
  over every surviving group carries a `lead_delta` for rank 0 on 91.4%
  of sightings where the harness's top-n series has one on 40.7%, and
  the displaced reading's fall is defined on 37.6%. On that all-group
  history the motion does not improve trust. Fitted on the validation
  split and scored on the testing split it moves the gap's rank-AUC by
  -0.0008 (interval -0.0022 to +0.0006), the same against the gap
  recalibrated, and within a bucket of the gap it ranks at 0.46 to 0.56.
  At the README's bar it decides 8,040 of 9,230 correctly where the
  recalibrated gap decides 8,083 and the README's own rule 7,998;
  against the recalibration the exchange is 22 clips won to 65 lost,
  exact McNemar 4.3e-06. The figures are
  `[[rr:Six rules on the same clips, every one fitted off them]]` and
  `[[rr:What the motion figures say]]`. What buys the trust is the
  recalibration, which needs no field, and the readings' entropy, which
  is a function of the confidences one partial already carries.
- The earlier reading of this, a gain over the README's rule, was a
  selection effect. The harness derived the motion from the readings it
  was shown, so a delta existed only where rank 0 was already in the
  reported list at the previous advance: the two fifths of sightings
  where the contest was old enough to be visible, which are not the
  sightings a host needs the field for. Read off the runtime's history,
  the gain is gone.
- `lead_delta` is kept on the states evidence and not the trust
  evidence: the reads it supports are a word coming, silence being kept
  and a lead that is being lost,
  `[[rr:TD-9#Every reading names its relation to the partial]]`.
- One advance later a revision has already shown on 97% of the clips
  that have an advance left; on one-second clips that advance is the
  last on 85% of them, so the split measures the wait for the final,
  not the motion.
- Motion persists only while a contest is live: consecutive lead
  velocities correlate at 0.41 within 1 to 4 nats on the clips and
  0.21 on the built stream, not at all past 4, where a won contest
  stops moving. A reading's share of the beam's softmax showed no such
  persistence (0.03), being bounded and saturating once a word has won,
  which is why the key is the pairwise lead and not the share. At the
  sighting the leader's velocity is large by selection, it has just
  overtaken, and says little; one advance later it separates,
  survivors gaining about 1.8 nats an advance and revised readings
  reversing by about 5. Only one reading in eighty lives three advances
  on a one-second clip, so the clip figures rest on few samples; the
  built stream gives the same band and shows momentum is not one
  phenomenon: the empty reading's lead persists (0.45) where an
  extending reading's barely does (+0.09) and a differing one's reverts
  (-0.23), which is what `relation`
  is for when reading `lead_delta`. A host wanting a longer window sums
  successive deltas of one reading across advances, reading each
  advance once and treating a null as a broken history, not a zero.
- Candidate preview is a real, weak signal. On the built stream the
  coming word's reading was in the beam before rank 0 grew on 52% of
  transitions, typically one advance earlier, and at that lead the top
  preview named the right word one time in six; this holds regardless
  of whether the acoustic onset was foreseen, and whether motion or mere
  presence supplies it is not established. Spliced isolated words; to be
  measured on continuous recordings.
- Whether motion foretells a word is unproven. On the built stream an
  `extends` reading gaining called 83% of the words, but 58% of those
  calls came before any sample of the coming word's clip had been fed
  and carry no acoustic evidence of it, so the lead they are made by is
  a read on the silence and not a warning of the word. Against the
  cheaper reads at the same false alarm rate, with each call matched to
  at most one word, the motion is the best of them (F1 0.61 against
  0.49 for the same candidate merely existing, 0.51 for its lead, 0.55
  for the trailing silence span and 0.54 for a clock the host keeps),
  and the figures move with the pause lengths the stream was built
  from. Exploratory, on spliced isolated words; the read is not
  recommended until it is measured on continuous recordings. Kept silence is the empty reading
  leading with a lead that hovers, 4 to 5 nats, at a velocity near
  zero. As an end-of-speech signal the motion trades rather than wins,
  so `[[rr:TD-8#Decision outcome]]` stands.
- Per-word readiness, correct dispatch and the consumer's acceptance
  are not established by any of this; they need a host policy measured
  on the consumer's continuous recordings and configuration.
- A reading's history is per utterance; `lead_delta` is null on the
  first advance after an endpoint.

## Implemented by

- `[[rr:update_readings]]`: the history over every surviving group and
  the traced top n, one grouping per chunk decoded.
- `[[rr:grouped]]`: the grouping the trace and the alternatives are both
  taken from.
- `[[rr:census_counts]]` and `[[rr:process_emitting]]`: the merge and
  reading-loss counters the deferred lattice question is measured with.
- `[[rr:trace_groups]]`, read by `stream --trace-groups`: the diagnostic
  trace the runtime's fields are checked against.
- `[[rr:advance_states]]`, with
  `[[rr:scripts/partial_trust.py#runtime_motion]]` and
  `[[rr:scripts/partial_trust.py#check_motion]]`: the harness's trace of
  the series, reading the runtime's two keys off the partial and
  checking them against its own derivation where both are defined.
