# Model Card: Selective GraphSAGE Risk Intelligence for KinetixRAG

**Version:** 1.0.0-draft  
**Last Updated:** 2026-10-17  
**Model Family:** Inductive Graph Neural Network (GraphSAGE with SAGEConv)  
**Task:** Impending Industrial Machinery Anomaly & Failure Prediction  
**License:** MIT  

---

## 1. Intended Use & Domain Boundaries

### 1.1 Intended Use
- **Primary Function:** Predict imminent equipment failure or ISO 10816-3 Zone C anomaly progression over a designated look-ahead horizon ($\Delta t = 24\text{h}$) from fused vibration, thermal, and kinematic features.
- **Role in Architecture:** **Strictly Advisory Intelligence**. Provides an auxiliary machine-learning score alongside KinetixRAG's deterministic rules and GraphRAG knowledge evidence.
- **Intended Users:** Reliability engineers, maintenance operators, and plant monitoring supervisors.

### 1.2 Out-of-Scope & Prohibited Uses
- **No Autonomous Payment Authorization:** The model score MUST NOT autonomously trigger Lightning satoshi settlements or authorize financial disbursements.
- **No SCADA / Safety Interlock Replacement:** The model is not a SIL-rated safety instrumented system (SIS) and must not be wired to emergency shutdown valves.
- **No Financial Fraud Detection:** Although GraphSAGE implementation patterns were adapted from GNN fraud architectures, this model is exclusively designed for industrial telemetry and rotating machinery.

---

## 2. Industrial Data Provenance & Feature Contract

### 2.1 Training Data Source
- **Primary Industrial Dataset:** NASA Prognostics Center of Excellence (PCoE) Run-to-Failure Bearing Dataset (University of Cincinnati IMS Test 2, Bearing 1 Outer Race Failure).
- **Sampling Parameters:** 20 kHz sampling rate, 2000 RPM speed, 6000 lbs radial load, PCB 353B33 accelerometer.
- **Progression Ground Truth:** Verified progression from healthy break-in operation ($1.85\text{ mm/s}$) to catastrophic outer raceway failure ($11.75\text{ mm/s}$).

### 2.2 Feature Contract (`v1.0.0`)
Ordered feature vector of dimension $D = 7$:
1. `vibration_rms_mm_s`: Overall vibration velocity RMS (mm/s) according to ISO 10816-3.
2. `vibration_peak_mm_s`: Peak vibration amplitude.
3. `vibration_crest_factor`: Peak-to-RMS ratio indicating impact severity / micro-spalling.
4. `vibration_kurtosis`: Statistical fourth moment of acceleration signals indicating impulsive faults.
5. `bearing_temp_c`: Bearing drive-end temperature in Celsius.
6. `temp_rate_of_change_c_per_hr`: Thermal derivative indicating friction runaway.
7. `frequency_dominant_hz`: Dominant spectral peak frequency identifying shaft harmonics vs. ball-pass frequencies (BPFO/BPFI).

### 2.3 Graph Construction & Anti-Leakage Topology
- **Nodes:** Timestamped sensor telemetry observations ($t_i$).
- **Edges:** Strictly causal forward chronological edges (`TEMPORAL_PRECEDES`) where $t_{\text{source}} < t_{\text{target}}$.
- **Prohibition:** No edges originate from future observations or post-failure states into earlier evaluation windows.

---

## 3. Label Definition & Dataset Splitting

### 3.1 Target Label Semantics
- **Binary Target ($y \in \{0, 1\}$):**
  - $y = 0$: Nominal operation / permissible degradation (ISO Zone A/B).
  - $y = 1$: Impending failure event occurring within the next 24 hours (ISO Zone C/D excursion or severe defect).

### 3.2 Temporal Splitting Strategy
- Data is strictly partitioned chronologically using `SplitStrategy.TEMPORAL_HOLDOUT`:
  - **Train Split:** Early run-to-failure lifecycle (baseline and initial stable runs).
  - **Validation Split:** Intermediate degradation period (used for early stopping and hyperparameter selection).
  - **Test Split:** Late lifecycle and critical failure onset (held out, evaluated once).
- **Leakage Invariant:** $\max(T_{\text{train}}) < \min(T_{\text{val}}) < \min(T_{\text{test}})$. Zero index overlap.

---

## 4. Evaluation & Calibration Principles

- **Primary Evaluation Metric:** Precision-Recall AUC (PR-AUC) to address class imbalance in industrial failure forecasting.
- **Secondary Metrics:** ROC-AUC, F1-Score at operational threshold, Precision, Recall, Confusion Matrix.
- **Calibration Status:** `calibrated: false`. Raw sigmoid outputs represent uncalibrated relative risk rankings and must not be described as true physical probabilities without Platt scaling or isotonic calibration.

---

## 5. Explainability & Provenance Constraints

- **Methodology:** Bounded $k$-hop subgraph extraction ($k = 2$) and node influence attribution.
- **Safety Semantics:** Influence weights represent statistical model attribution only, not physical causation.
- **Degradation:** If explainability calculation times out or fails, the API safely returns `status: "timeout"` without interrupting predictive operations.

---

## 6. Operational & Deployment Guardrails

- **Default State:** `ENABLE_GRAPH_RISK_MODEL=false`.
- **Runtime Isolation:** PyTorch and PyTorch Geometric dependencies are isolated and lazy-loaded to protect memory-constrained environments (e.g. Render free tier 512 MB).
- **Fallback:** In the event of model loading or inference failure, KinetixRAG falls back gracefully to its verified deterministic ISO 10816-3 risk engine (`telemetry/predictive_intelligence.py`).
