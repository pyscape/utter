## Reading

Every figure on this page is an observation about where a word was, and none of
them names a cause. The three measurements are read as follows.

**The word among the readings at the sighting** is whether the clip's own word
led any reading the beam held at the first partial that carried a word. It does
not say the word was decodable then: the sighting is early in the utterance and
the audio that settles the word may not have been fed yet, which is what the
revision split below is about. utter's partial carries a ranked n-best of the
beam's surviving groups, `[[rr:TD-2#The decoder: partial alternatives]]`;
the stock wheel measured here exposes no such call, so its partial is one
reading, it has no rate above n = 1, and the two engines are comparable at
n = 1 only. Absence at the sighting is not evidence of
pruning, and presence below rank 0 is not evidence of a mis-scored cost; both are
the position, not the reason.

**The word among the readings in the final** at n = 1 is the accuracy the
Speech Commands page reports, arrived at the same way. Above n = 1 it is the
depth of the n-best, which the two engines reach differently: utter's is the
beam's own n-best and libvosk's is read off a lattice whose width is
`lattice-beam`, a setting this page leaves at the model's stock value in both
columns. A word at rank 3 is not a word a host could have had: nothing here says
which of the n to choose, and no figure on this page is a selection rule.

**Neither oracle rate is a ceiling.** The rate at the wider search is what this
search kept under this grammar at this beam, and reading it as the limit of the
acoustic model would be reading a search result as a model property. It bounds
nothing.

**The clips a wider search moves** is search sensitivity. A final that changes
when the beam is widened tells you the answer was not settled by a margin the
stock beam covers; it does not establish that the stock search erred, and a clip
that became right is not a clip the stock search should have got. The split into
became-right, became-wrong and wrong-to-wrong is there because a wider search
moves clips in both directions, and a net accuracy difference hides that.

**The range of beams** is there because one wider setting cannot tell an
insensitive final from a lucky pair of settings. It is counted on a sample of
the split rather than all of it, and it counts finals that differ, not finals
that are wrong. A row of zeros says the final under this grammar does not depend
on the beam over that range; it says nothing about a larger grammar, about
another model, or about the partials along the way, which are not compared here.

**That the sibling directory is read** is established by a third sibling, not
assumed from the beam figures: libvosk prints the beam and max-active it parsed
as it opens a model, which the page quotes, and both engines are then shown to
return different finals when a key with an undoubted effect is changed. Without
that, a row of zeros would be indistinguishable from a configuration file that
was never opened.

**The compute column** is one steady-state pass per setting on the same joined
audio, the pass the Speech Commands page uses, so the two columns are comparable
with each other and with that page. It is a single run on one machine, not a
minimum over repeats, so read the p50 and treat small differences in the tails as
noise.

**The revision split** is where the word that replaced a first word stood when
that first word appeared, read off a recorded series. It decodes nothing, so it
inherits that series' terms: a record's word is the first word a host was shown,
and the replacing word is taken as the word leading the clip's last advance,
which is the last partial a host saw and not always the final the engine emitted
(the page gives the rate at which those two agree). Revisions whose sighting was
also the clip's last advance are set aside rather than bucketed: the series holds
no later reading to place the replacing word by.

Of the buckets, one has a lever already named elsewhere and one has a lever with
a price. A replacing word that stood among the readings at the sighting but below
rank 0 is a question about cost, where the graph scale and the bigram cost sit,
and `[[rr:TD-6]]` settled that the runtime reproduces libvosk's choices there
rather than improving on them, so the lever exists and parity is what it costs.
A revision the next advance settles is the audio arriving at the decoding
cadence, and the only thing that moves the cadence is the chunk
(`[[rr:TD-2#Inputs: configuration]]`), which is read from `conf/model.conf` and
is part of the byte identity the runtime is gated on. The remaining buckets have
no lever named on this page and none is guessed for them.
