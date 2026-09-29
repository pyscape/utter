## Reading

**Certainty is what the acoustic model heard, not whether the word is right.**
Misread spoken words sit only a little below right ones, since the model heard
speech either way. What it tells apart is speech from a room: every word decoded
out of a background recording scored below the lowest tenth of the right ones.
That is 44 room words over six recordings at three floors, so the 100% it rejects
is a count on this page and not a rate to carry elsewhere.

**Energy over the floor does nearly as well on these recordings**, and taking
both does not reject more room words here, because the room words are already
all rejected. What certainty adds is a figure that needs no floor.

**A threshold on certainty depends on the input level.** Turning a clip down
lowers its certainty, by a seventh at -20 dB and a fifth at -30 dB in the median,
while the grammar hardly moves it. A host that sets a threshold sets it for the
level it runs at.

**The sound outside words keeps the kinds of sound apart** on synthetic sounds
laid over a synthetic room: steady noise raises `broad`, a hum raises `low`, an
impulse shows as a short rise, and the example rule sorts every such window. The
recorded dishes, a mix of clatter and running water, fall between the kinds, and
three room windows read as impulses; the rule is the page's, and a host writes its
own. The rise starts a frame before the impulse's own sample in the median, which
is the frame grid and not a lead.
