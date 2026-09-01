# Rating instructions — pass "sentiment_r"

You are rating short texts. Rate each text ON ITS OWN, one at a time,
independently. Do not compare texts to each other, do not try to guess
where any text came from, and do not skip any text.

## Dimension

SENTIMENT of the described events, integer 1-7.
1 = very negative events, 4 = neutral or mixed, 7 = very
positive events. Score what happens in the text, not the
quality of the writing.

## Input / output

Each `batch_NN.json` file in this folder is a JSON array of
`{"id": ..., "text": ...}` objects. For every batch file, write
`scores/batch_NN.scores.json` containing a JSON array with one entry
per input id, in the same order:

    [{"id": "<id>", "score": <integer 1-7>}, ...]

No other keys, no commentary, no markdown fences in the output files.
