Operator notes (NOT for the judge session):

- Give the judge ONE pass directory at a time (e.g. `scene/`).
  Never this folder, `key.json`, the repository, or the project
  name.
- Run each pass contiguously in fresh sessions; record the
  judge's visible model/version string and the date per pass in
  PROVENANCE.md.
- The judge writes `scores/batch_NN.scores.json` inside the pass
  directory. Validate with `python analyze_study2.py judging\claude`.
