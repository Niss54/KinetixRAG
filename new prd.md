# PRD: Selective GraphSAGE Risk Intelligence for KinetixRAG

**Status:** Implementation specification  
**Date:** 2026-10-17  
**Primary repository:** https://github.com/Niss54/KinetixRAG  
**Reference repository:** https://github.com/markasame/fraud-detection-gnn  
**Target branch:** `feat/graphsage-risk-intelligence`

---

## 1. Executive Summary

Add a carefully scoped, optional graph-based risk intelligence capability to KinetixRAG by reusing only the transferable engineering patterns from `fraud-detection-gnn`: GraphSAGE model structure, training safeguards, graph construction patterns, imbalanced-class handling, evaluation methodology, and graph explainability.

KinetixRAG is an industrial telemetry and GraphRAG platform, not a financial fraud detection application. Do **not** copy the reference repository wholesale, reuse its Elliptic Bitcoin model weights, or treat its transaction features and labels as industrial data.

KinetixRAG already contains predictive and telemetry logic, including:

- `telemetry/predictive_intelligence.py`
- `telemetry/generator.py`
- `telemetry/pattern_match.py`
- `telemetry/draft.py`
- `backend/app/api/telemetry.py`
- `frontend/app/predictive-watch/page.tsx`

The new capability must complement the existing deterministic health/risk workflow, not duplicate or replace it blindly. The existing deterministic approach and a suitable non-graph baseline must be retained for comparison. GraphSAGE must not become an operational source of risk scores until suitable industrial features, labels, evaluation results, and runtime compatibility are verified.

**Delivery rule:** build and debug in phases, run the required regression checks, and push the implementation branch to GitHub only after all required checks pass. Never hide failures by deleting tests, weakening assertions, or silently skipping coverage.

## 2. Problem Statement

KinetixRAG correlates telemetry with equipment, failure events, work orders, procedures, and supporting documents. Its existing predictive logic is primarily deterministic/rule-based. A graph ML model may add value if related telemetry windows and relevant historical context help forecast impending failure better than the current approach or a non-graph model.

The reference fraud repository demonstrates useful GNN engineering, but its dataset and target task do not match industrial maintenance. This implementation must prove that graph structure helps the industrial task before presenting the model as an improvement.

## 3. Goals

1. Audit whether available KinetixRAG industrial data supports a defensible, reproducible graph-learning task.
2. Reuse only generic, relevant techniques from the reference repository.
3. Define a stable industrial feature/label contract and a causal, leakage-aware graph representation.
4. Compare a non-graph baseline and GraphSAGE on the same industrial samples and splits.
5. Integrate an optional, lazy-loaded inference path into the existing FastAPI backend.
6. Show model availability, score provenance, explanation status, and limitations in the existing Predictive Watch experience.
7. Preserve the current telemetry, GraphRAG, work-order, agent, and machine-money behaviour.
8. Test, debug, document, and push the completed work only after acceptance gates pass.

## 4. Non-Goals

- Do not import or implement a financial transaction fraud product in KinetixRAG.
- Do not copy the reference repository's Streamlit dashboard or stand-alone FastAPI app.
- Do not replace Neo4j/GraphRAG retrieval with a GNN.
- Do not delete or rewrite existing telemetry and risk-scoring functionality just to make the model fit.
- Do not add an ML score as an automatic payment-authorization or machine-money settlement condition.
- Do not claim production predictive accuracy from synthetic data, a smoke test, or the reference repository's reported baseline.
- Do not commit raw datasets, large model checkpoints, credentials, generated caches, or unrelated project assets.

## 5. Current-State Constraints to Verify at the Start

Before editing code, inspect the current branch and confirm the following against the checked-out source. Repository structure may have changed since this PRD was written.

- Existing risk calculation: `telemetry/predictive_intelligence.py`.
- Existing telemetry generation and signal contract: `telemetry/generator.py` and related telemetry modules.
- Existing signature matcher and warning logic: `telemetry/pattern_match.py`, `telemetry/draft.py`.
- Existing HTTP integration: `backend/app/api/telemetry.py`, `backend/app/main.py`.
- Existing operator page: `frontend/app/predictive-watch/page.tsx` and its telemetry components/API client.
- Existing tests: `tests/telemetry/`, `tests/backend/`, `tests/retrieval/`, `tests/agents/`; inspect the actual test paths before running commands.
- The README references NASA IMS bearing telemetry. Verify whether usable raw samples and their provenance are present locally, how they are transformed into online telemetry, and whether the same feature schema can be used for training and inference.
- The top-level requirements aggregate several subsystem requirement files. Do not blindly add the reference repository's entire pinned dependency set.
- The backend contains a comment about model pre-warming and memory limits on a free deployment tier. Treat PyTorch memory and cold-start impact as a release risk.

Record the actual starting commit, `git status`, test baseline, discovered data sources, and any existing failures in `docs/GRAPHSAGE_INTEGRATION_AUDIT.md` before implementation.

## 6. Reference Code Reuse Policy

### Reuse or adapt only the following ideas

| Reference file | What may be reused/adapted | What must be replaced |
|---|---|---|
| `common.py` | GraphSAGE module pattern, explicit model configuration, checkpoint loading discipline, k-hop explanation concepts | Elliptic labels, Bitcoin IDs, fixed input dimensions, source-specific paths and label names |
| `build_graph.py` | Graph tensor construction, train-only preprocessing, temporal split discipline, validation of mappings and indices | Elliptic downloader, Bitcoin transaction graph, Elliptic feature mapping and label rules |
| `train_gnn.py` | Reproducible seeds, class weighting, validation-driven early stopping, best-checkpoint saving | Source-specific target, split, dimensions, and hard-coded training assumptions |
| `baseline.py` | Same-samples/same-split baseline comparison principle | Financial dataset-specific setup and unmodified target assumptions |
| `evaluate.py` | Precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, persisted report pattern | Source-specific plots/paths and assumptions that the GNN must win |
| `explain.py` | GNNExplainer and bounded k-hop subgraph explanation pattern | Bitcoin transaction IDs, illicit/licit colors and labels, source-specific visualizations |
| `app.py` | API contract and one-time/lazy model loading patterns as reference only | Do not copy a second FastAPI application; add a router/service inside KinetixRAG |

### Do not copy as product code

- `streamlit_app.py`
- The full reference `requirements.txt` pins
- Reference Docker configuration or compose file without an actual deployment need
- Elliptic dataset/model artifacts, transaction IDs, and reported metrics as KinetixRAG results
- Unrelated reference UI and data-download scripts

If any non-trivial code is adapted, preserve MIT attribution and add a concise entry to `THIRD_PARTY_NOTICES.md` containing the source project, URL, license, and files/logic adapted. Keep the reference licence and notices where required.

## 7. Product and Architecture Requirements

### FR-01: Preserve existing behaviour

All existing telemetry endpoints, health reports, GraphRAG retrieval, work-order drafts, event flows, tests, and machine-money controls must retain their contracts and behaviour unless a directly relevant test proves a defect that must be fixed. No unrelated refactors.

### FR-02: Establish data readiness before model training

The audit must identify a real, domain-appropriate industrial training source and a defensible target label. Prefer an existing, provenance-documented NASA IMS dataset already used/referenced by KinetixRAG, if it can support the selected task. Verify source, data terms, sampling units, and any preprocessing already used by the repository.

If an adequate industrial dataset or label cannot be established, do not train a purported industrial model on arbitrary generated labels. Complete the disabled/experimental integration contract and tests, clearly mark live ML inference unavailable, document the blocker, and do not claim the model is trained or validated. Synthetic telemetry may be used for deterministic unit/integration tests only, not for real-world efficacy claims.

### FR-03: Define a causal graph and feature contract

Document the graph task in `docs/GRAPHSAGE_INTEGRATION_AUDIT.md` or a linked model card before model implementation:

- Unit of prediction, for example a timestamped telemetry window or equipment observation.
- Exact target semantics, positive class, prediction horizon, and any excluded/ambiguous windows.
- Feature names, units, windowing, missing-value handling, normalisation, and schema version.
- Node/edge meanings and how edges are created from verified chronology/equipment relationships.
- Data provenance and train/validation/test membership.

Only include information available at the prediction timestamp. Do not use future telemetry windows, future work orders, future labels, or post-failure data to create the current sample's features or neighbours. Split by time/run/equipment as appropriate to the real-world generalisation claim. Fit scalers and feature selection on training data only. Use a graph projection fit for the task; do not feed every heterogeneous Neo4j node into a homogeneous `SAGEConv` model without a defined feature mapping and rationale.

### FR-04: Build a comparable non-graph baseline

Implement a reproducible non-graph baseline on the exact same labelled industrial samples, features, and split as GraphSAGE. Use the existing deterministic KinetixRAG health/risk logic as an operational comparator where comparable, but do not pretend unlike scores are directly comparable without defining the evaluation method.

At minimum, report PR-AUC, precision, recall, F1, ROC-AUC, confusion matrix, class prevalence, and chosen operating threshold. Prioritise PR-AUC for imbalanced failure labels. Record dataset version, split boundaries, feature schema version, random seed, software versions, and model/checkpoint version.

### FR-05: Implement GraphSAGE as an optional model

- Keep model, feature preparation, training, checkpoint loading, and inference in a small dedicated module, proposed path `ml/graph_risk/`.
- Make dimensions, model parameters, paths, and thresholds explicit configuration, not hidden constants distributed across files.
- Use deterministic seed configuration and validation-based early stopping.
- Save checkpoint metadata: architecture version, feature schema version, source dataset, label definition, training timestamp, metrics, and dependency/model version.
- Validate input dimensions, non-finite values, missing features, unknown entities, empty graphs, and incompatible checkpoints with explicit errors.
- Do not load arbitrary pickle/checkpoint files from untrusted sources. Load only locally generated and validated artefacts.
- No source model weights from the fraud repository may be used for industrial predictions.

### FR-06: Add explainability with honest semantics

Where supported, use GNNExplainer or a simpler deterministic explanation to return the most influential connected samples/features. Bound the size and latency of explanations. Label explanation weights as model influence, not proof of causation. If an explanation fails or the feature is unavailable, return an explicit status rather than a fabricated explanation.

### FR-07: Integrate within the current backend

Use the existing FastAPI app and add a narrowly scoped router/service, proposed files:

- `ml/graph_risk/` for model code and model-level tests
- `backend/app/services/graph_risk.py`
- `backend/app/api/graph_risk.py`
- corresponding tests under the repository's established test layout

Register the router in `backend/app/main.py` using the existing conventions. Reuse existing Pydantic/FastAPI patterns. Do not copy `app.py` into the repository as a separate application.

Proposed endpoints, subject to existing route conventions:

- `GET /api/graph-risk/status`: enabled/disabled state, checkpoint readiness, model/schema version, and validation status.
- `POST /api/graph-risk/predict`: validated industrial input or telemetry-window reference; returns advisory model score only when a compatible, validated model is loaded.

Responses must distinguish `available`, `disabled`, `model_not_trained`, `incompatible_features`, and `inference_error`. Never report a made-up probability. Only call a value a probability if calibration has been assessed; otherwise return a clearly named score and `calibrated: false`.

### FR-08: Protect operational and payment safety

- The current deterministic health checks and GraphRAG evidence path remain operational if the optional model is disabled, unavailable, times out, or fails to load.
- ML output is advisory. It must not independently issue work orders, shut equipment down, or authorize/execute a payment.
- The existing human approval and machine-money policy checks must not be weakened or bypassed.
- A model failure must degrade to a documented safe state, not crash the whole backend or block unrelated API routes.

### FR-09: Runtime/dependency isolation

- Put ML-specific dependencies in `ml/graph_risk/requirements.txt` or another clearly scoped optional dependency file.
- Do not merge all pinned dependencies from the fraud repository into KinetixRAG's base requirements.
- Keep PyTorch/torch-geometric imports lazy or isolated so the existing API can start with ML disabled.
- Measure RAM, cold-start time, inference latency, and checkpoint size on the intended runtime. If the current deployment cannot safely run PyTorch within its memory budget, keep the model out of the main process and document a separate optional service boundary as a later/explicit deployment choice.
- Never commit `.env`, API keys, private certificates, datasets, model weights, caches, or generated binary output unless the repository already explicitly versions an appropriately sized/test fixture.

### FR-10: Frontend integration

Integrate into the existing Predictive Watch/telemetry UI. Do not create a Streamlit app or an unrelated new frontend. Show:

- model status and model/schema version;
- advisory score, only when available;
- the prediction timestamp and the feature/data window used;
- concise explanation/contributing context where supported;
- a visible label that the ML score is experimental/advisory until validated;
- a useful disabled/error/unsupported-input state;
- the existing deterministic risk and GraphRAG evidence as separate fields, not silently blended into one number.

The UI must remain usable when the model is disabled or the backend has no model weights.

## 8. Proposed File Layout

Follow repository conventions if the audit finds a better location; do not create duplicate functionality.

```text
ml/
  graph_risk/
    __init__.py
    schemas.py
    features.py
    graph_builder.py
    model.py
    train.py
    inference.py
    explain.py
    evaluate.py
    requirements.txt
    README.md
    tests/
backend/app/
  api/graph_risk.py
  services/graph_risk.py
tests/
  ... existing test layout ...
docs/
  GRAPHSAGE_INTEGRATION_AUDIT.md
  GRAPH_RISK_MODEL_CARD.md
THIRD_PARTY_NOTICES.md
```

Only create files needed by the validated design. Do not generate empty wrappers just to match this tree.

## 9. Phased Build Plan and Acceptance Gates

Every phase must end with its stated gate. Fix failures before progressing. Keep implementation commits scoped and preserve a clean rollback path.

### Phase 0: Repository safety and baseline capture

Tasks:
1. Inspect `git status`, current branch, latest commit, and remote configuration.
2. Create `feat/graphsage-risk-intelligence` from the intended base branch. Do not work directly on `main`.
3. Run the existing tests/build checks before changes; save exact command output and identify pre-existing failures.
4. Inspect current telemetry risk logic, API/client contract, Neo4j relationships, deployment constraints, tests, and dataset/provenance.
5. Create `docs/GRAPHSAGE_INTEGRATION_AUDIT.md` with architecture notes, reuse decision table, data readiness, dependency/runtime risks, test baseline, and an explicit GO / LIMITED GO / NO-GO recommendation for operational model activation.

Acceptance gate:
- Existing code and data paths are understood.
- Baseline checks and any existing failures are recorded.
- No application runtime code has been changed before the audit is complete.
- Any unavailable dataset/license/label issue is explicit.

### Phase 1: Data and task definition

Tasks:
1. Verify data provenance and availability. Prefer relevant existing industrial telemetry over unrelated Elliptic transactions.
2. Select one clear prediction target and label rule; define its time horizon and intended use.
3. Define a versioned feature schema with units and deterministic transformations.
4. Define a graph projection that only uses information available at prediction time.
5. Define group/temporal splits that avoid future leakage and overly optimistic random-node splitting.
6. Add a small, hand-verified test fixture with documented synthetic or sampled-data provenance. Do not include a large raw dataset in Git.

Acceptance gate:
- A reviewer can reproduce how one labelled sample and its edges are created.
- Train/validation/test boundaries are fixed, documented, and leakage-aware.
- Feature, node, edge, and label semantics are explicit.
- If valid industrial labels cannot be established, mark the model as not trainable/validated and proceed only with safe disabled-mode/API-contract work.

### Phase 2: Feature pipeline and graph builder

Tasks:
1. Implement deterministic feature extraction with unit tests for missing, non-finite, constant, malformed, and out-of-range inputs.
2. Implement graph construction with validated node-ID-to-index mapping, explicit edge direction, and graph shape checks.
3. Verify no duplicate or out-of-range indices, invalid edges, mismatched features/labels, or preprocessing leakage.
4. Persist only compact derived metadata needed to reproduce training; keep large source datasets outside Git.
5. Add tests for empty graph, isolated node, one-node graph, disconnected components, and missing optional history.

Acceptance gate:
- Same input produces the same features/graph under a fixed configuration.
- All graph indices and tensor shapes validate.
- Future information cannot enter an earlier prediction sample.
- No production input silently receives an arbitrary default feature vector.

### Phase 3: Non-graph baseline and evaluation harness

Tasks:
1. Establish a simple baseline, preferably XGBoost or a defensible existing deterministic comparator, on the same feature matrix and split as GraphSAGE.
2. Handle class imbalance without using test labels to tune weights or thresholds.
3. Save metric output as machine-readable JSON plus a readable report.
4. Report PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix, class prevalence, threshold, and split/data provenance.
5. Add reproducibility checks and a smoke test that runs without downloading datasets or requiring paid APIs.

Acceptance gate:
- Baseline is reproducible and evaluated on held-out data.
- The report identifies dataset version and label/split semantics.
- There are no claims that the GNN outperforms the baseline before the GNN has been evaluated.

### Phase 4: GraphSAGE training and checkpointing

Tasks:
1. Adapt the architecture pattern from `common.py` and training safeguards from `train_gnn.py`, keeping the industrial input schema explicit.
2. Use fixed seeds, train-only preprocessing, validation-driven early stopping, and class weighting only where justified.
3. Save the best validation checkpoint with schema/model metadata; do not track bulky model binaries by default.
4. Implement strict checkpoint compatibility checks and a clear unavailable-model state.
5. Run the test split once after training choices are final. Do not tune against test results.

Acceptance gate:
- Training finishes on the supported environment or a documented reason explains why the candidate model cannot be trained there.
- Checkpoint reload gives the same predictions within defined numerical tolerance.
- The training report contains baseline-versus-GNN metrics on identical samples/splits.
- If GraphSAGE does not improve on the baseline or does not justify its latency/memory cost, it stays experimental and must not replace the current risk pipeline.

### Phase 5: Explanation and result provenance

Tasks:
1. Add a bounded GNNExplainer/k-hop explanation where the graph/model supports it.
2. Return contributing sample/entity identifiers and features only when those identifiers can be mapped back safely.
3. Add timeout/failure handling and tests for explanation unavailable, isolated nodes, and oversized neighborhoods.
4. Write `docs/GRAPH_RISK_MODEL_CARD.md` covering intended use, data provenance, labels, known limitations, evaluation, calibration, and deployment state.

Acceptance gate:
- Explanations never assert causality and never invent entities.
- Prediction and explanation can fail independently without taking down telemetry.
- Model card clearly separates benchmark results from production claims.

### Phase 6: Backend service and API integration

Tasks:
1. Add `backend/app/services/graph_risk.py` and `backend/app/api/graph_risk.py` (or the audited equivalent).
2. Register the router using current conventions in `backend/app/main.py`.
3. Keep ML dependencies lazy/optional and set a documented configuration flag such as `ENABLE_GRAPH_RISK_MODEL=false` by default until validation and deployment checks pass.
4. Add status and predict contracts with Pydantic validation, safe error codes, versioned schema, and request-size/feature-count bounds.
5. Ensure API responses include model state, score type/calibration status, timestamp, model version, feature schema version, and limitations.
6. Ensure errors do not leak traceback, local file paths, or secret configuration.
7. Keep existing endpoint inputs/outputs unchanged.

Acceptance gate:
- Existing backend starts with the ML feature disabled and without importing heavy ML packages unnecessarily.
- `/api/graph-risk/status` works in disabled, trained/available, and model-missing states.
- Invalid inputs return a defined client error; inference errors return a safe degraded response.
- Existing telemetry and machine-money routes remain functional.

### Phase 7: Frontend integration

Tasks:
1. Reuse the existing Predictive Watch page/components and API client; do not duplicate the application.
2. Add a compact graph-risk panel/section that distinguishes deterministic risk, GraphRAG evidence, and ML output.
3. Implement loading, success, disabled, unavailable, invalid input, timeout, and error states.
4. Add component tests for each state and verify responsive layout/accessibility conventions used by the project.
5. Do not show fake model numbers when the service is disabled or the model has not passed evaluation.

Acceptance gate:
- UI builds and tests pass.
- Disabled/unavailable mode remains useful and does not break existing telemetry controls.
- The ML status and score provenance are visible and understandable.

### Phase 8: Full testing and regression suite

Run the actual scripts found in the repository. Expected checks include the following when applicable:

```bash
# Backend/core regression suite
pytest tests/backend tests/retrieval tests/agents tests/telemetry -q

# Full Python suite, if supported by the repository
pytest -q

# Python syntax/import checks for changed modules
python -m compileall ml/graph_risk backend/app/api/graph_risk.py backend/app/services/graph_risk.py

# Frontend unit/component tests and lint/build
cd frontend
npm run test
npm run lint
npm run build

# Browser e2e suite, if the local/browser environment is available
npm run test:e2e
```

Do not assume every path above exists: inspect the repository and run the equivalent correct commands. Run the relevant tests from the repository root where their import configuration expects it. Do not install unnecessary packages or download a large dataset just to run a unit test.

Add coverage for:
- current deterministic health/risk outputs remain unchanged;
- feature-schema validation and invalid/missing values;
- graph construction, edge mapping, split integrity, and leakage checks;
- model training smoke test on a tiny test graph;
- checkpoint save/reload and incompatible checkpoint rejection;
- model disabled/not trained/available/error states;
- API validation and response schema;
- explanation success and failure handling;
- frontend disabled, error, and available states;
- machine-money authorization cannot be triggered by an ML score;
- regression of existing Predictive Watch and telemetry flows.

Acceptance gate:
- Every required suite is green.
- Any skipped test has a documented, legitimate environmental reason; no test is skipped merely to get a pass.
- No secrets, model binaries, datasets, screenshots, caches, or unrelated generated files are added accidentally.

### Phase 9: Debugging and hardening loop

If any build, test, type check, lint, API, or integration check fails:

1. Capture the complete failing command and the first meaningful traceback/assertion.
2. Reproduce the failure with the narrowest relevant test.
3. Identify and fix the root cause; do not merely suppress the exception, relax assertions, or change a test to match broken behaviour.
4. Rerun the focused test, then the related module suite.
5. Rerun all required backend tests, frontend tests/lint/build, and relevant end-to-end tests after the fix.
6. Inspect test coverage and `git diff` for regressions, accidental edits, generated files, secrets, and license/attribution issues.
7. Repeat this loop until all required checks pass. Record each material issue and fix in the implementation notes/PR summary.

If a check is genuinely blocked by credentials, unavailable infrastructure, unsupported hardware, or missing licensed data, document the evidence and stop short of claiming full completion. Do not label the phase as passed and do not push a branch described as complete while required checks are failing or unresolved.

Acceptance gate:
- All locally runnable required checks pass after the last code change.
- No known failing required test remains.
- Any environmental blocker is explicit and the affected feature remains safely disabled.

### Phase 10: Final review and documentation

Tasks:
1. Update the relevant README/architecture docs only where necessary; avoid rewriting unrelated judge/Wispr evidence.
2. Document installation of optional ML dependencies, data preparation/training commands, evaluation output, feature flag, model status, API schema, and rollback/disable procedure.
3. Add/update the model card and third-party notice.
4. Report the actual baseline-versus-GNN results, not expected results or reference-repository metrics.
5. Run a final `git diff --check`, inspect the full diff, scan for secrets, and rerun relevant suites after documentation/build config changes.
6. Confirm the app still works with `ENABLE_GRAPH_RISK_MODEL=false` and with missing checkpoint/model dependencies.

Acceptance gate:
- Implementation and operational docs match the actual code.
- No unsupported performance claims remain.
- Final diff includes only scoped, relevant changes.

### Phase 11: Commit and push to GitHub

Do not begin this phase until Phase 8 through Phase 10 acceptance gates pass.

1. Verify the active branch is `feat/graphsage-risk-intelligence` (or a clearly named equivalent) and `git status` contains only intended changes.
2. Run the final required test commands again after the last code change.
3. Run `git diff --check` and inspect `git diff --stat` plus the full diff.
4. Stage only relevant source, tests, docs, and dependency changes. Never stage `.env`, keys, tokens, raw datasets, local model artifacts, caches, or unrelated user changes.
5. Commit with a clear message, for example: `feat: add optional graph risk intelligence`.
6. Push the feature branch to the configured GitHub remote. Do not force-push and do not push directly to `main`.
7. If GitHub authentication/remote permissions are available, verify the remote branch and commit SHA after push. Open a pull request only if the environment/tooling supports it; do not merge without explicit authorization.
8. If authentication or network access prevents pushing, report that honestly and provide the exact local branch name, commit SHA (if committed), failed command, and manual push command. Never claim that a push succeeded without verifying it.

Acceptance gate:
- All required checks passed before the push.
- Intended commit is present on the remote feature branch, or a precise external blocker is reported without claiming success.

## 10. Quality, Performance, and Release Criteria

The work is complete only when all applicable criteria below are satisfied:

- Existing KinetixRAG core flows and regression suites pass.
- Industrial data provenance and label semantics are documented and reproducible.
- Train/validation/test splits and graph construction are leakage-aware.
- Baseline and GraphSAGE are evaluated on identical samples and splits.
- Model results are presented accurately; no claim of GNN superiority is made without evidence.
- The GNN is not activated as the primary operational predictor unless it adds demonstrated value and passes resource checks. Otherwise it remains clearly experimental/disabled.
- Model dependencies are optional/isolated, and base application startup remains reliable without them.
- No ML score can bypass human review or machine-money policy controls.
- API and UI handle unavailable model, invalid input, timeout, and explanation failures safely.
- Documentation, model card, and license attribution are complete.
- All required tests/lint/build checks pass after the final source change.
- GitHub push is completed only after verification, with no secrets or unrelated files included.

## 11. Final Agent Report Template

At completion, report:

1. **Decision:** operational GraphSAGE enabled, experimental/disabled, or blocked, with evidence.
2. **Files:** exact files added/modified and why each was needed.
3. **Reuse:** which concepts/files from the reference repo were adapted and attribution location.
4. **Data:** source, label semantics, split strategy, feature schema, and known limitations.
5. **Metrics:** actual baseline and GraphSAGE evaluation results, or a clear reason metrics could not be produced.
6. **Tests:** exact commands and pass/fail counts for backend, model, frontend, build, and e2e tests.
7. **Runtime:** dependency strategy, memory/latency findings, and feature-flag state.
8. **Git:** branch name, commit SHA, remote push verification, and PR URL if one was actually opened.
9. **Blockers:** anything unresolved, without describing the work as fully complete.
