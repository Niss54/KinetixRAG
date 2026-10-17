# GraphSAGE Risk Intelligence: Architectural Audit & Integration Assessment

**Date:** 2026-10-17  
**Repository:** `https://github.com/Niss54/KinetixRAG`  
**Base Commit:** `32a68657536a3f5d09e0e43c4c98eb1799a346cb`  
**Active Branch:** `feat/graphsage-risk-intelligence`  
**Author / Auditor:** Antigravity (Governed by Gravitas Reliability Engine)  
**Status:** COMPLETE (Phase 0 Baseline)

---

## 1. Executive Baseline & Repository Safety

This audit establishes the pre-implementation baseline for selectively integrating GraphSAGE risk intelligence into KinetixRAG, in accordance with `new prd.md`.

### 1.1 Git Status & Branch Discipline
- **Source Branch:** `main` (clean working directory, tracking `origin/main`).
- **Feature Branch:** `feat/graphsage-risk-intelligence` created cleanly from `32a6865`.
- **Remote Configuration:** `origin -> https://github.com/Niss54/KinetixRAG.git` (fetch/push verified).
- **Branch Rule:** All development and commits will occur exclusively on `feat/graphsage-risk-intelligence`. No direct commits to `main`.

### 1.2 Baseline Test Suite Verification
- **Python / Pytest Baseline:**
  - Command: `py -3.12 -m pytest tests/ -q`
  - Output: `335 passed, 1 warning in 235.61s`
  - Zero pre-existing test failures across all backend, agent, retrieval, telemetry, and machine-money modules.
- **Frontend / Vitest Baseline:**
  - Command: `npm run test` (in `frontend/`)
  - Output: `16 test files passed, 61 passed in 25.53s`
  - Zero failures across UI component, accessibility, and client API suites.
- **Frontend Production Build Baseline:**
  - Command: `npm run build` (in `frontend/`)
  - Output: Compiled successfully with Turbopack, TypeScript passed (13.2s), 12/12 routes prerendered/generated.
- **Combined Pre-Implementation Baseline:**
  - Total tests verified: **396 passed, 0 failed**.
  - All application runtime paths and baseline suites are 100% green.

---

## 2. Architectural Analysis of Existing Telemetry & Risk Subsystems

KinetixRAG already contains an established, deterministic industrial telemetry and predictive intelligence engine:

### 2.1 Predictive Intelligence (`telemetry/predictive_intelligence.py`)
- **Governing Standard:** ISO 10816-3 Class II Rotating Machinery (pumps/bearings).
- **Thresholds:**
  - `vibration_de`: Nominal 2.5 mm/s, Warning 6.5 mm/s, Critical 10.0 mm/s.
  - `temperature_de`: Nominal 65.0 °C, Warning 85.0 °C, Critical 105.0 °C.
- **Composite Risk Calculation:**
  $$\text{Composite Risk} = 0.40 \cdot \text{VibRisk} + 0.35 \cdot \text{TempRisk} + 0.25 \cdot \text{MaintenancePenalty}$$
- **Health Index:** $\text{HealthIndex} = 100 \cdot (1.0 - \text{CompositeRisk})$.
- **Action Windows:** `IMMEDIATE_SHUTDOWN`, `PLAN_WITHIN_72_HOURS`, `SCHEDULE_NEXT_TURNAROUND`, `NORMAL_MONITORING`.
- **Deterministic Work Order Drafts:** Generates structured maintenance suggestions (`PM02` high priority).

### 2.2 Telemetry Ingestion & Scanning (`backend/app/api/telemetry.py`)
- Endpoints:
  - `POST /api/telemetry/scan`: Accepts `equipment_tag`, simulates drift, matches failure patterns via Neo4j / cosine similarity, and creates predictive events.
  - `POST /api/telemetry/draft`: Emits draft work orders requiring operator review.

### 2.3 Operator Console (`frontend/app/predictive-watch/page.tsx`)
- Integrates `TelemetryPanel` with live signal sliders and deviation gauges.
- Highlights the core thesis: **"Why GraphRAG over a simple SCADA threshold?"** — PLC triggers symptom alerts (e.g. 5.42 mm/s vibration), while GraphRAG verifies warranties, historical work orders, and SOPs before authorizing expenditures.
- **Safety Gate:** Machine-money settlements require operator sign-off or strictly enforced policy caps. ML models must NEVER independently disburse funds.

---

## 3. Industrial Data Readiness & Provenance

### 3.1 Discovered Public Dataset: NASA IMS Bearing Run-to-Failure
- **Location:** `telemetry/fixtures/nasa_ims_bearing_sample.json`
- **Adapter:** `telemetry/adapters/public_dataset.py` (`PublicDatasetReplayAdapter`)
- **Institution:** University of Cincinnati / NASA Prognostics Center of Excellence (PCoE).
- **Metadata:**
  - Test 2: Bearing 1 Outer Race Failure.
  - Sampling Rate: 20 kHz, Speed: 2000 RPM, Radial Load: 6000 lbs.
  - Sensor: PCB 353B33 High Frequency Accelerometer.
  - Progression Stages: `HEALTHY_BASELINE` -> `NOMINAL_STABLE` -> `INITIAL_DEGRADATION` -> `ZONE_C_ANOMALY_EXCURSION` -> `CRITICAL_FAILURE`.
- **Contract Compatibility:** Normalized to `TelemetryEventSchema` with `vibration_mm_s`, `bearing_temp_c`, `frequency_peak_hz`, `vibration_rms_g`, and `iso_zone`.
- **Conclusion:** Excellent real industrial dataset fixture with authentic run-to-failure progression and zero financial/crypto leakage. Ideal for Phase 1 feature and graph construction.

---

## 4. Reference Repository Reuse Decision Table

Reference repository: `https://github.com/markasame/fraud-detection-gnn`

| Reference Component | Transferable Engineering Pattern | Required Adaptation / Replacement | Decision |
|---|---|---|---|
| `common.py` | SAGEConv layer structure, explicit config dataclasses, checkpoint save/load discipline | Replace Elliptic Bitcoin node schemas with industrial vibration/thermal schemas. Parameterize dimensions. | **ADAPT** |
| `build_graph.py` | PyG `Data` tensor assembly, train-only scaler fitting, temporal masking | Replace Bitcoin transaction edge logic with causal sensor-equipment temporal topology. Ensure anti-leakage. | **ADAPT** |
| `train_gnn.py` | Deterministic seed setting, class weighting, validation-loss early stopping, best checkpoint saving | Bind to industrial binary failure horizon ($T + \Delta t$). | **ADAPT** |
| `baseline.py` | Evaluation of non-graph tabular model on identical split | Train LogisticRegression/RandomForest on identical features. | **ADAPT** |
| `evaluate.py` | PR-AUC, ROC-AUC, F1, confusion matrix calculation, JSON persistence | Replace fraud plots with industrial risk classification reports. | **ADAPT** |
| `explain.py` | GNNExplainer / k-hop influence subgraph extraction | Bound latency, fallback on timeout, explain industrial sensor nodes. | **ADAPT** |
| `app.py` | FastAPI route patterns and lazy model loading | Do NOT copy application. Integrate cleanly into `backend/app/api/graph_risk.py`. | **ADAPT ROUTER** |
| `streamlit_app.py` | Streamlit dashboard | Do NOT import. Surface UI in existing Next.js `PredictiveWatchPage`. | **REJECT** |
| Bitcoin Datasets | Elliptic transaction features and illicit labels | Completely irrelevant to pumps/turbines. | **REJECT** |
| Pinned Dependencies | Heavy pinned requirements.txt | Isolate optional dependencies in `ml/graph_risk/requirements.txt`. | **REJECT GLOBAL PIN** |

---

## 5. Deployment Constraints & Runtime Isolation

1. **Free Tier RAM Budget (Render 512 MB):**
   - Comment in `backend/app/main.py`: Pre-warming PyTorch / heavy transformers causes kernel OOM termination.
   - **Isolation Rule:** ML models must be lazy-loaded on demand. The FastAPI application must boot in `< 1.0s` without loading PyTorch when `ENABLE_GRAPH_RISK_MODEL=false`.
2. **Current Python Environment:**
   - Available: `scikit-learn`, `numpy`, `scipy`, `pydantic`, `fastapi`.
   - Optional ML: `torch`, `torch_geometric` to be isolated in `ml/graph_risk/requirements.txt`.
3. **Operational Safety Rule:**
   - ML output is strictly advisory. Response schema returns `calibrated: false` and `advisory: true`.
   - Cannot trigger autonomous Lightning payments or bypass human approval.

---

## 6. Official Recommendation

### Recommendation: **LIMITED GO (Advisory / Experimental)**

- **Rationale:** Industrial dataset provenance (NASA IMS) is verified and locally available. Tabular and deterministic baselines exist. However, due to deployment memory constraints and model unvalidated operational status, GraphSAGE MUST remain an optional, lazy-loaded, advisory capability behind a default-off feature flag (`ENABLE_GRAPH_RISK_MODEL=false`).
- **Gate Status:** Phase 0 Acceptance Criteria satisfied. Proceed to Phase 1 data contract definition.
