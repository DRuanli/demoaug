# Changelog

## v1.0.0 — Initial release

First public release accompanying the manuscript submission.

### Features

- 4-stage filter cascade for fair text augmentation (label fidelity → probe calibration → diversity → MMD)
- IH_class-based hardness measure with KL-divergence Hard-bias
- Pareto independence analysis between Hard-bias_IH and ABROCA
- Six fairness/utility metrics: AUC, ABROCA, Hard-bias_IH, DI gap, EO gap, Calibration gap
- Statistical rigor: paired t, Cohen's d, TOST equivalence, Bayes factor, bootstrap correlation CI
- Two-regime experimental design: DistributionalBias and LabelNoiseBias
- Toy encoder + toy LLM for reproducible CPU-only experiments
- BERT and Mistral-7B integration paths (untested in v1, marked for real-data run)
- 17 unit tests covering all math-critical modules

### Known limitations

See `docs/07_limitations.md` for the full list. Highlights:

- Real-data validation is missing (synthetic dataset only)
- Toy bigram LLM is much weaker than Mistral
- No human evaluation of generated text
- No ethics analysis section
- ABROCA is the headline fairness metric (other metrics reported but not central)
