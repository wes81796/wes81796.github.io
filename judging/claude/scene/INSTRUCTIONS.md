# Rating instructions — pass "scene"

You are rating short texts. Rate each text ON ITS OWN, one at a time,
independently. Do not compare texts to each other, do not try to guess
where any text came from, and do not skip any text.

## Dimension

NARRATIVITY, integer 1-7.
1 = pure summary / telling: events reported abstractly, no scene.
7 = fully dramatized scene: moment-by-moment action, quoted
dialogue, a specific located moment in time.
Ignore how pleasant or unpleasant the described events are.
Ignore how much sensory detail is present. A flat summary of a
catastrophe is still a 1; a moment-by-moment scene of a calm
afternoon can be a 7.

## Input / output

Each `batch_NN.json` file in this folder is a JSON array of
`{"id": ..., "text": ...}` objects. For every batch file, write
`scores/batch_NN.scores.json` containing a JSON array with one entry
per input id, in the same order:

    [{"id": "<id>", "score": <integer 1-7>}, ...]

No other keys, no commentary, no markdown fences in the output files.
