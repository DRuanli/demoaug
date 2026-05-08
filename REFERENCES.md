# References

This document lists all the references the DEMOAUG paper builds on, organized by **how directly they shape the contribution**. Each entry has the full citation, verified DOI, and a 1-2 sentence note on why it matters for this project.

The references are organized in four tiers:
- **Tier 1**: Core foundations the paper directly extends or critiques
- **Tier 2**: Methodological building blocks (algorithms and metrics we implement)
- **Tier 3**: Closely related augmentation and fairness work
- **Tier 4**: Broader context (ethics, practitioner needs, fairness theory)

---

## Tier 1 — Core foundations (direct extensions / direct critiques)

These are the references the paper most directly builds on. Reading these is essential for understanding the contribution.

### Sha, Gašević & Chen (2023) — primary baseline (extended)
> Sha, L., Gašević, D., & Chen, G. (2023). Lessons from debiasing data for fair and accurate predictive modeling in education. *Expert Systems with Applications*, 228, 120323.
>
> **DOI**: [10.1016/j.eswa.2023.120323](https://doi.org/10.1016/j.eswa.2023.120323)

The paper this work directly extends. Sha et al. propose Method-3 (kDN-based instance-hardness selection) for fair augmentation in educational text classification. Our pipeline replaces the SMOTE step with constrained LLM generation, replaces kDN with IH_class, and empirically tests whether their implicit Hard-bias → ABROCA chain holds. Our Forum-Post synthetic dataset mirrors their Table 1 cell counts (1413/812/925/553).

### Smith, Martinez & Giraud-Carrier (2014) — IH_class definition
> Smith, M. R., Martinez, T., & Giraud-Carrier, C. (2014). An instance level analysis of data complexity. *Machine Learning*, 95(2), 225–256.
>
> **DOI**: [10.1007/s10994-013-5422-z](https://doi.org/10.1007/s10994-013-5422-z)

Defines instance hardness as ensemble miss-rate `IH_class(z, y) = 1 - (1/|L|) Σ P_l(y | z)` over a learner ensemble. Our `src/hardness.py` implements this with `{LR, SVM-RBF, RandomForest}`. We use IH_class as the principled high-dimensional alternative to Sha et al.'s kDN, then derive Hard-bias_IH as the symmetric KL between per-group IH distributions.

### Chawla, Bowyer, Hall & Kegelmeyer (2002) — SMOTE baseline
> Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
>
> **DOI**: [10.1613/jair.953](https://doi.org/10.1613/jair.953)

The 30,000+ citation algorithm we critique on geometric grounds (off-manifold phantom embeddings, ~3 nats below real-text density in our PCA-whitened KDE) but find empirically robust on ABROCA via implicit regularization. Our Section 4 finding — that SMOTE's off-manifold property is not necessarily harmful — is one of the paper's central empirical results.

### Gardner, Brooks & Baker (2019) — ABROCA metric
> Gardner, J., Brooks, C., & Baker, R. (2019). Evaluating the Fairness of Predictive Student Models Through Slicing Analysis. In *Proceedings of the 9th International Conference on Learning Analytics and Knowledge (LAK '19)* (pp. 225–234).
>
> **DOI**: [10.1145/3303772.3303791](https://doi.org/10.1145/3303772.3303791)

Defines ABROCA (Absolute Between-ROC Area) as `∫ |ROC_g0(t) - ROC_g1(t)| dt`. This is the headline fairness metric Sha et al. 2023 (and our paper) uses. Our central empirical finding — that ABROCA is statistically uncorrelated with Hard-bias_IH — directly engages with how this metric behaves under augmentation.

---

## Tier 2 — Methodological building blocks

These are the algorithms and tests our pipeline implements as components.

### Devlin, Chang, Lee & Toutanova (2019) — BERT encoder
> Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186).
>
> **DOI**: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)

`bert-base-uncased` is the production encoder in `src/encoder.py` (gated by `USE_REAL_MODELS=True`). All real-data runs use BERT [CLS] embeddings. The toy TF-IDF+SVD encoder is a CPU-friendly stand-in for v1 experiments; BERT is the default for the published-paper run described in `docs/06_real_data_migration.md`.

### Jiang et al. (2023) — Mistral 7B generator
> Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., ... El Sayed, W. (2023). Mistral 7B. *arXiv preprint*.
>
> **DOI**: [10.48550/arXiv.2310.06825](https://doi.org/10.48550/arXiv.2310.06825)

The production LLM in `src/llm_gen.py` (`MistralLLM` class). We use Mistral-7B-Instruct-v0.3 with HuggingFace `LogitsProcessor` for hard token banning (logit = -∞ on banned subword IDs). This is strictly stronger than instruction-level negative prompting (which has 30-40% violation rate on 7B-class models). The Apache 2.0 license makes Mistral suitable for educational research deployment.

### Gretton, Borgwardt, Rasch, Schölkopf & Smola (2012) — MMD test
> Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A Kernel Two-Sample Test. *Journal of Machine Learning Research*, 13(25), 723–773.
>
> **JMLR URL**: [https://jmlr.org/papers/v13/gretton12a.html](https://jmlr.org/papers/v13/gretton12a.html)
>
> **ACM DL DOI**: [10.5555/2188385.2188410](https://doi.org/10.5555/2188385.2188410)

Implements the Maximum Mean Discrepancy permutation test we use as Filter C4 (`src/filter_mmd.py`). RBF kernel with median pairwise-distance bandwidth heuristic, 200 permutations, p > 0.05 acceptance criterion. Note: the unbiased MMD² estimator can go slightly negative under H₀ (a sampling artifact); a unit test in our suite caught this and we now short-circuit with p=1 when observed MMD² ≤ 0.

### Yu, Zhuang, Zhang, Meng, Ratner, Krishna, Shen & Zhang (2023) — AttrPrompt (LLM data generation framework)
> Yu, Y., Zhuang, Y., Zhang, J., Meng, Y., Ratner, A., Krishna, R., Shen, J., & Zhang, C. (2023). Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets & Benchmarks Track*.
>
> **arXiv DOI**: [10.48550/arXiv.2306.15895](https://doi.org/10.48550/arXiv.2306.15895)

The most relevant prior work on LLM-based training-data generation. AttrPrompt shows that diverse attributed prompts beat simple class-conditional prompts and reveal LLM regional biases. Our work extends this insight to fairness-specific augmentation: instead of just *diverse* attributes, we use *V_disc-banned* attributes to remove demographic shortcuts while preserving cell authenticity.

---

## Tier 3 — Closely related augmentation & fairness work

These are works in the same research thread (fair augmentation for educational/general ML) that we compare against, build on, or position our contribution relative to.

### Sha, Raković, Das, Gašević & Chen (2022) — earlier class-balancing work
> Sha, L., Raković, M., Das, A., Gašević, D., & Chen, G. (2022). Leveraging Class Balancing Techniques to Alleviate Algorithmic Bias for Predictive Tasks in Education. *IEEE Transactions on Learning Technologies*, 15(4), 481–492.
>
> **DOI**: [10.1109/TLT.2022.3196278](https://doi.org/10.1109/TLT.2022.3196278)

The precursor paper to Sha 2023. Establishes that class balancing can reduce ABROCA on educational tasks and motivates the Method-3 design refined in the 2023 paper. Important context for understanding why our extension targets the 2023 framework specifically rather than this earlier version.

### Yan, Kao & Ferrara (2020) — Fair Class Balancing
> Yan, S., Kao, H., & Ferrara, E. (2020). Fair Class Balancing: Enhancing Model Fairness without Observing Sensitive Attributes. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM '20)* (pp. 1715–1724).
>
> **DOI**: [10.1145/3340531.3411980](https://doi.org/10.1145/3340531.3411980)

Demonstrates clustering-based fair class balancing that does not require sensitive-attribute observation. Methodologically adjacent to our work (uses cluster proxies for sensitive groups), but addresses a different problem setting (when A is unobserved). Useful for the related-work section to position DEMOAUG in the broader fair-balancing landscape.

### Borchers & Baker (2025) — ABROCA distributional analysis
> Borchers, C., & Baker, R. S. (2025). ABROCA Distributions For Algorithmic Bias Assessment: Considerations Around Interpretation. In *Proceedings of the 15th International Learning Analytics and Knowledge Conference (LAK '25)*.
>
> **DOI**: [10.1145/3706468.3706498](https://doi.org/10.1145/3706468.3706498)

Recent (2025) statistical analysis of ABROCA's distributional properties under varying AUC differences and class imbalance. Directly relevant to our Cohen's d / TOST / Bayes factor analysis since it characterizes when ABROCA differences are practically meaningful vs sampling noise. We cite their finding that ABROCA exhibits high skewness as motivation for our equivalence-margin choices.

---

## Tier 4 — Broader context (ethics, practitioner needs, fairness theory)

These are the references that contextualize the work in the broader ML-fairness and educational-technology landscape.

### Holstein, Wortman Vaughan, Daumé III, Dudík & Wallach (2019) — practitioner needs
> Holstein, K., Wortman Vaughan, J., Daumé III, H., Dudík, M., & Wallach, H. (2019). Improving Fairness in Machine Learning Systems: What Do Industry Practitioners Need? In *Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems (CHI '19)* (pp. 1–16).
>
> **DOI**: [10.1145/3290605.3300830](https://doi.org/10.1145/3290605.3300830)

Empirical study of 35 interviews + 267 practitioner survey on what fairness tools are actually needed in industry. Cited in our `docs/07_limitations.md` as motivation for building inspectable LLM augmentation (vs SMOTE's opaque phantom embeddings) — practitioners explicitly request auditability of training data, which DEMOAUG provides.

### Suresh & Guttag (2021) — sources-of-harm framework
> Suresh, H., & Guttag, J. (2021). A Framework for Understanding Sources of Harm Throughout the Machine Learning Life Cycle. In *Proceedings of the 1st ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO '21)* (Article 17, pp. 1–9).
>
> **DOI**: [10.1145/3465416.3483305](https://doi.org/10.1145/3465416.3483305)

The standard taxonomy of seven harm sources in ML pipelines (historical, representation, measurement, aggregation, learning, evaluation, deployment). DEMOAUG specifically targets *representation bias* (under-sampled cells) and *aggregation bias* (group-vocabulary signals). Cited in `docs/07_limitations.md` for the ethics-section commitment.

---

## Citation conventions used in the codebase

When code comments reference these works, we use the format `(Author, Year)` followed by the DOI in a longer comment block. Examples:

```python
# IH_class definition (Smith, Martinez & Giraud-Carrier 2014).
# DOI: 10.1007/s10994-013-5422-z
def fit_and_score(self, Z, Y): ...
```

```python
# MMD permutation test (Gretton et al. 2012, JMLR 13:723-773).
# Median-heuristic bandwidth as standard.
def mmd_permutation_test(X, Y, n_perm): ...
```

```python
# Sha et al. 2023 Table 1 cell counts (Forum, sex attribute).
# DOI: 10.1016/j.eswa.2023.120323
base_counts = {(0, 1): 1413, (0, 0): 812, (1, 1): 925, (1, 0): 553}
```

This makes the lineage of every algorithmic decision traceable.

---

## A note on access

All references in this document have **publicly accessible** versions either through the publisher (if open-access) or via author-deposited preprints on arXiv / the authors' websites. We have linked DOIs (which redirect to publisher pages) rather than direct PDFs since publisher pages are stable while authors' personal pages move.

For Sha et al. 2023 (Tier 1) the *Expert Systems with Applications* version is open-access under Creative Commons. For BERT, Mistral, and AttrPrompt the arXiv versions are open. The Smith et al. 2014 *Machine Learning* paper requires institutional access; Springer often grants it through national consortium agreements.

If you cannot access a reference and need it for reproducing this work, please open an issue on the repository and we will discuss alternatives.

---

## How to extend this list

When future revisions add references (e.g., for the human-evaluation component, ethics analysis, or real-data validation), maintain the four-tier structure:

1. Add the entry under the appropriate tier based on directness of relevance.
2. Include the verified DOI (use [doi.org](https://doi.org/) lookup or ACM/Springer/IEEE indices).
3. Write 1-2 sentences explaining why it matters specifically for DEMOAUG — not just what the cited paper does. The "why for us" framing is what makes this document useful as a reading guide rather than just a bibliography.
