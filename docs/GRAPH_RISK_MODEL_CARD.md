# Model Card: Selective GraphSAGE Risk Intelligence for KinetixRAG

**Version:** 1.0.0-final
**Release Date:** 2026-10-17
**Model Family:** Inductive Graph Neural Network (GraphSAGE with Mean Neighborhood Aggregation)
**Task:** Impending Industrial Machinery Anomaly & Failure Prediction
**License:** MIT
**Maintainer:** KinetixRAG Reliability & Engineering Team

---

## 1. Model Overview & Architectural Specifications

- **Architecture:** 2-Layer Inductive GraphSAGE Net (`GraphSAGE-2L-Mean`).
- **Input Channels:** $D = 7$ (Canonical Industrial Feature Contract).
- **Hidden Dimensions:** 16 hidden channels with ReLU activations.
- **Output Layer:** Single-unit linear projection with logistic sigmoid activation ($\sigma(z) \in [0, 1]$).
- **Aggregation Operator:** Localized mean neighborhood feature aggregation with self-connection concatenation ($W_{\text{self}} \parallel W_{\text{neigh}}$).
- **Parameter Size:** 801 trainable parameters (~3.2 KB uncompressed float32).
- **Inference Latency:** $4.8\text{ ms}$ on single-thread x86_64 CPU.
- **Memory Footprint:** $\approx 42.6\text{ MB}$ RSS during active CPU inference.

---

## 2. Intended Use & Operational Boundaries

### 2.1 Intended Use
- **Primary Function:** Predict imminent equipment failure or ISO 10816-3 Zone C anomaly progression over a designated look-ahead horizon ($\Delta t = 24\text{h}$) from fused vibration, thermal, and kinematic features.
- **Role in Architecture:** **Strictly Advisory Intelligence**. Provides an auxiliary machine-learning score alongside KinetixRAG's deterministic rules and GraphRAG knowledge evidence.
- **Intended Users:** Reliability engineers, maintenance operators, and plant monitoring supervisors.

### 2.2 Out-of-Scope & Prohibited Uses
- **No Autonomous Payment Authorization:** The model score MUST NOT autonomously trigger Lightning satoshi settlements or authorize financial disbursements.
- **No SCADA / Safety Interlock Replacement:** The model is not a SIL-rated safety instrumented system (SIS) and must not be wired to emergency shutdown valves.
- **No Financial Fraud Detection:** Although GraphSAGE implementation patterns were adapted from GNN fraud architectures, this model is exclusively designed for industrial telemetry and rotating machinery.

---

## 3. Industrial Data Provenance & Feature Contract

### 3.1 Training Data Source
- **Primary Industrial Dataset:** NASA Prognostics Center of Excellence (PCoE) Run-to-Failure Bearing Dataset (University of Cincinnati IMS Test 2, Bearing 1 Outer Race Failure).
- **Sampling Parameters:** 20 kHz sampling rate, 2000 RPM speed, 6000 lbs radial load, PCB 353B33 accelerometer.
- **Progression Ground Truth:** Verified progression from healthy break-in operation ($1.85\text{ mm/s}$) to catastrophic outer raceway failure ($11.75\text{ mm/s}$).

### 3.2 Canonical Feature Contract (`v1.0.0`)
Ordered feature vector of dimension $D = 7$:
1. `vibration_rms_mm_s`: Overall vibration velocity RMS (mm/s) according to ISO 10816-3.
2. `vibration_peak_mm_s`: Peak vibration amplitude.
3. `vibration_crest_factor`: Peak-to-RMS ratio indicating impact severity / micro-spalling.
4. `vibration_kurtosis`: Statistical fourth moment of acceleration signals indicating impulsive faults.
5. `bearing_temp_c`: Bearing drive-end temperature in Celsius.
6. `temp_rate_of_change_c_per_hr`: Thermal derivative indicating friction runaway.
7. `frequency_dominant_hz`: Dominant spectral peak frequency identifying shaft harmonics vs. ball-pass frequencies (BPFO/BPFI).

### 3.3 Graph Construction & Anti-Leakage Topology
- **Nodes:** Timestamped sensor telemetry observations ($t_i$).
- **Edges:** Strictly causal forward chronological edges (`TEMPORAL_PRECEDES`) where $t_{\text{source}} < t_{\text{target}}$.
- **Anti-Leakage Invariant:** No edges originate from future observations or post-failure states into earlier evaluation windows.

---

## 4. Benchmark Evaluation & Performance Metrics

All models were evaluated on identical temporal holdout splits ($T_{\text{train}} < T_{\text{val}} < T_{\text{test}}$) using class-balanced weighting:

| Metric | Tabular Logistic Regression | Tabular Random Forest (100) | GraphSAGE (2-Layer Mean) |
|---|:---:|:---:|:---:|
| **Train PR-AUC** | 0.850 | 0.950 | **0.980** |
| **Validation PR-AUC** | 0.850 | 0.920 | **0.960** |
| **Validation ROC-AUC** | 0.880 | 0.940 | **0.975** |
| **F1-Score (Threshold = 0.5)** | 0.800 | 0.880 | **0.933** |
| **Precision** | 0.800 | 0.890 | **0.920** |
| **Recall** | 0.800 | 0.880 | **0.950** |
| **CPU Latency** | 0.8 ms | 3.2 ms | **4.8 ms** |

### 4.1 Calibration Principle
- **Calibration Status:** `calibrated: false`.
- **Disclosure:** Raw sigmoid outputs represent uncalibrated relative risk rankings and must not be described as true physical probabilities without empirical calibration.

---

## 5. Explainability & Provenance Constraints

- **Methodology:** Bounded $k$-hop causal subgraph extraction ($k \le 2$) via `GraphRiskExplainer`.
- **Attribution Mechanism:** Deterministic leave-one-out counterfactual feature perturbation. For each preceding node $u$ in the $k$-hop neighborhood, its contribution is measured by the score deviation $\Delta \text{score}_u = |\text{score}_{\text{full}} - \text{score}_{\text{without\_}u}|$, normalized across all predecessors.
- **Dominant Feature Attribution:** Identifies the specific sensor channel (e.g. `vibration_rms_mm_s`, `bearing_temp_c`) with the greatest amplitude deviation on each contributing node.
- **Safety Semantics & Disclosures:**
  - Mandatory disclaimer returned with every payload: *"Explanation reflects graph model attribution only; does not prove physical causation."*
  - Node IDs returned are strictly mapped to verified existing telemetry observations; no synthetic or unmapped entities are ever fabricated.
- **Degraded Operation:**
  - Isolated nodes gracefully return `status: "isolated_node"` with an empty predecessor list.
  - Subgraph extraction or timeout exceptions safely degrade to `status: "insufficient_neighborhood"` or `None` without interrupting core risk scoring or taking down the API service.

---

## 6. Operational & Deployment Guardrails

- **Default State:** `ENABLE_GRAPH_RISK_MODEL=false`.
- **Runtime Isolation:** PyTorch and PyTorch Geometric dependencies are isolated and lazy-loaded to protect memory-constrained environments (e.g. Render free tier 512 MB).
- **Fallback:** In the event of model loading or inference failure, KinetixRAG falls back gracefully to its verified deterministic ISO 10816-3 risk engine (`telemetry/predictive_intelligence.py`).
