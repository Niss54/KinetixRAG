# Third-Party Notices and Open Source Attribution

KinetixRAG incorporates open-source software and architectural concepts subject to the terms and conditions of their respective licenses. This document details third-party projects, academic contributions, and reference repositories utilized or adapted during the development of KinetixRAG.

---

## 1. Reference Architecture Attribution: `fraud-detection-gnn`

KinetixRAG's inductive GraphSAGE module (`ml/graph_risk/`) adapts architectural and training pipeline concepts from the open-source reference repository:

- **Project:** `fraud-detection-gnn`
- **Source:** [https://github.com/markasame/fraud-detection-gnn](https://github.com/markasame/fraud-detection-gnn)
- **License:** MIT License
- **Copyright:** (c) 2024 markasame

### Architectural Concepts Adapted:
- **Two-Layer Mean Neighbor Aggregation:** Formulation of inductive localized neighborhood embedding concatenation with self-features.
- **Class-Imbalanced Binary Cross-Entropy:** Inverse-prevalence positive class weighting during gradient optimization.
- **Validation-Driven Early Stopping:** Epoch-level validation loss and PR-AUC monitoring with best weight checkpoint restoration.
- **Train-Only Feature Scaler Fitting:** Standardizing numerical feature distributions strictly on training indices to prevent future statistical leakage.

### Explicit Replacements & Scope Boundaries (PRD Compliance):
- **Domain Replacement:** All cryptocurrency transaction schemas, Elliptic Bitcoin graph features, and illicit transaction labels were **completely excluded**. The module operates exclusively on industrial ISO 10816-3 rotating machinery telemetry (vibration RMS, peak, crest factor, kurtosis, temperature, rate of change, dominant frequency).
- **Anti-Leakage Temporal Graph:** Replaced transaction-flow edges with strict causal temporal predecessor edges (`t_source < t_target`) preventing backward information propagation.
- **Governance Barrier:** Financial fraud scoring patterns were replaced with strictly advisory, uncalibrated operational risk scores (`calibrated: false`, `is_advisory: true`). The model is decoupled from machine-money settlement pipelines and cannot authorize fund transfers.
- **Modular Dependency Isolation:** Global pinned requirements were replaced with a lightweight, lazy-loaded implementation supporting CPU-only operation on 512 MB RAM environments without pre-warming overhead.

### MIT License Text (`fraud-detection-gnn`):
```text
MIT License

Copyright (c) 2024 markasame

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Academic Foundation: GraphSAGE

The inductive graph neural network architecture is founded on the seminal work by Hamilton et al.:

- **Paper:** *Inductive Representation Learning on Large Graphs*
- **Authors:** William L. Hamilton, Rex Ying, Jure Leskovec (Stanford University)
- **Conference:** Advances in Neural Information Processing Systems (NeurIPS 2017)
- **Reference:** Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS 2017*, pp. 1024-1034.

---

## 3. Industrial Dataset Provenance: NASA IMS Bearing Run-to-Failure

The empirical industrial telemetry fixture used for validation and inductive feature benchmarking originates from:

- **Dataset:** Intelligent Maintenance Systems (IMS) Bearing Run-to-Failure Dataset
- **Institutions:** University of Cincinnati (Center for Intelligent Maintenance Systems) and NASA Prognostics Center of Excellence (PCoE)
- **Source:** NASA Prognostics Data Repository
- **Experiment:** Test 2, Bearing 1 Outer Race Failure (20 kHz sampling rate, 2000 RPM, 6000 lbs radial load)
- **License / Terms:** Public Domain / Open Data for Research and Educational Use
- **Normalized Fixture Location:** `ml/graph_risk/tests/fixtures/industrial_telemetry_fixture.json`

---

## 4. Core Frameworks and Open Source Dependencies

| Package / Library | License | Primary Function |
|---|---|---|
| **Python** | PSF License | Runtime environment (3.12) |
| **FastAPI** | MIT License | High-performance asynchronous web framework |
| **Pydantic** | MIT License | Strict data validation and contract enforcement |
| **NumPy & SciPy** | BSD 3-Clause | Numerical and scientific array computation |
| **scikit-learn** | BSD 3-Clause | Tabular baselines (Random Forest, Logistic Regression) & PR-AUC evaluation |
| **PyTorch & PyTorch Geometric** | BSD-style / MIT | Optional inductive tensor computation and graph convolution |
| **Neo4j Driver** | Apache 2.0 | Cypher graph database communication |
| **Qdrant Client** | Apache 2.0 | Vector similarity search and dense embeddings |
| **Next.js & React** | MIT License | Operator dashboard and responsive user interface |
| **Lucide React** | ISC License | Industrial operator interface iconography |
| **Tailwind CSS** | MIT License | Utility-first CSS presentation |
