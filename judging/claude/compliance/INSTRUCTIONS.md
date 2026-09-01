# Rating instructions — pass "compliance"

You are rating short texts. Rate each text ON ITS OWN, one at a time,
independently. Do not compare texts to each other, do not try to guess
where any text came from, and do not skip any text.

## Dimension

For each text, list every DISTINCT concrete sensory detail about
the food (taste, smell, texture, temperature, appearance, sound),
then give the count. A detail must be a specific rendered
perception ("the sauce had gone gluey"), not a bare verdict
("the food was bad").

## Input / output

Each `batch_NN.json` file in this folder is a JSON array of
`{"id": ..., "text": ...}` objects. For every batch file, write
`scores/batch_NN.scores.json` containing a JSON array with one entry
per input id, in the same order:

    [{"id": "<id>", "details": ["<detail>", ...], "count": <integer>}, ...]

No other keys, no commentary, no markdown fences in the output files.
