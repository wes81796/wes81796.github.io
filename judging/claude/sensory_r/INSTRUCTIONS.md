# Rating instructions — pass "sensory_r"

You are rating short texts. Rate each text ON ITS OWN, one at a time,
independently. Do not compare texts to each other, do not try to guess
where any text came from, and do not skip any text.

## Dimension

SENSORY RICHNESS, integer 1-7.
Score the density of concrete sensory perception in the text:
taste, smell, texture, sound, sight, temperature, bodily
sensation. A vividly rendered disgusting taste scores exactly as
high as a vividly rendered delicious one: score the AMOUNT of
sensory rendering, not its appeal. Ignore whether the text is
story-like or summary-like.

## Input / output

Each `batch_NN.json` file in this folder is a JSON array of
`{"id": ..., "text": ...}` objects. For every batch file, write
`scores/batch_NN.scores.json` containing a JSON array with one entry
per input id, in the same order:

    [{"id": "<id>", "score": <integer 1-7>}, ...]

No other keys, no commentary, no markdown fences in the output files.
