## Reading

Every model opens in both engines, and each passes the three bars with
room: partials at 98.9% or better, word times at 98.1% or better,
endpoints at 96.6% or better. Segment equality is above the English
model's 94.15% in all four languages. That comparison is loose, because
a take here is a speaker saying one short word per clip with the
recording's own pauses between clips, and G2's corpus is continuous
command speech.

French is the weakest on every row. Its disagreement is split between
words only in libvosk's finals (975) and words only in the runtime's
(867), with few substitutions, which is the near-tie pattern G2 found
in English rather than one side inventing words. Against the prompts,
the two engines' word error is within 0.8 points in every language.

The Russian model has no Latin-script words, so the segment's Firefox
prompt is left out of its grammar; its clips are still decoded.

Two traps for anyone re-running this. The release's German TSV spells
the prompt "null" as "nan", a missing value written back out. The Speech
Commands harness builds its grammar with an ASCII-escaped JSON dump,
which the stock wheel does not unescape: it drops every non-ASCII word
with a warning, all of Russian and German "fünf" among them, while
utterpy reads the grammar correctly. The script sends the grammar
unescaped to both engines.
