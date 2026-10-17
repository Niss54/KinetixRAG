import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import GraphRiskWidget from "./GraphRiskWidget";
import * as api from "@/lib/api";

vi.mock("@/lib/api", () => ({
  getGraphRiskStatus: vi.fn(),
  predictGraphRisk: vi.fn(),
}));

describe("GraphRiskWidget (Phase 7: Predictive Watch UI Integration)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const mockDisabledStatus: api.GraphRiskStatusResponse = {
    status: "disabled",
    model_available: false,
    schema_version: "1.0.0",
    feature_flag_enabled: false,
    feature_names: [
      "vibration_rms",
      "vibration_peak",
      "crest_factor",
      "kurtosis",
      "temp",
      "temp_rate",
      "dom_freq",
    ],
    feature_dim: 7,
    detail: "Graph risk model is disabled via ENABLE_GRAPH_RISK_MODEL=false",
  };

  const mockAvailableStatus: api.GraphRiskStatusResponse = {
    status: "available",
    model_available: true,
    schema_version: "1.0.0",
    feature_flag_enabled: true,
    feature_names: [
      "vibration_rms",
      "vibration_peak",
      "crest_factor",
      "kurtosis",
      "temp",
      "temp_rate",
      "dom_freq",
    ],
    feature_dim: 7,
    checkpoint_path: "artifacts/checkpoints/best_graphsage.pt",
    device: "cpu",
  };

  const mockPredictResponse: api.GraphRiskPredictResponse = {
    schema_version: "1.0.0",
    sample_id: "SAMPLE-9999",
    equipment_id: "P-101A",
    advisory_risk_score: 0.742,
    predicted_class: 1,
    calibrated: false,
    is_advisory: true,
    deterministic_risk: 0.42,
    latency_ms: 12.5,
    explanation: {
      subgraph_node_count: 5,
      subgraph_edge_count: 8,
      hops_analyzed: 2,
      influential_nodes: [
        {
          node_id: "OBS-003",
          influence_score: 0.62,
          dominant_feature: "vibration_rms",
        },
        {
          node_id: "OBS-002",
          influence_score: 0.28,
          dominant_feature: "kurtosis",
        },
      ],
      caveat: "Attribution is bounded by causal 2-hop backward horizon and is advisory only.",
      status: "computed",
    },
  };

  it("renders disabled state with clear guidance banner when feature flag is off", async () => {
    vi.mocked(api.getGraphRiskStatus).mockResolvedValueOnce(mockDisabledStatus);

    render(<GraphRiskWidget equipmentId="P-101A" deterministicRisk={0.42} />);

    // Header and badges
    expect(screen.getByText("GraphSAGE Risk Intelligence")).toBeInTheDocument();
    expect(screen.getByText("Schema v1.0.0")).toBeInTheDocument();
    expect(screen.getByText("Advisory Only")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("Disabled")).toBeInTheDocument();
    });

    // Verify informational banner explaining deployment safety
    expect(screen.getByText(/Optional ML Intelligence Layer/i)).toBeInTheDocument();
    expect(screen.getByText(/ENABLE_GRAPH_RISK_MODEL=false/i)).toBeInTheDocument();

    // Verify 3 core cards
    expect(screen.getByText("Deterministic Rule Score")).toBeInTheDocument();
    expect(screen.getByText("42.0%")).toBeInTheDocument();
    expect(screen.getByText("GraphSAGE Advisory Score")).toBeInTheDocument();
    expect(screen.getByText("--")).toBeInTheDocument();
    expect(screen.getByText("Governance Barrier")).toBeInTheDocument();
    expect(screen.getByText("Payment Interlock Active")).toBeInTheDocument();
    expect(screen.getByText("ML score cannot authorize funds")).toBeInTheDocument();
  });

  it("renders available state and allows scoring with 2-hop explainability", async () => {
    vi.mocked(api.getGraphRiskStatus).mockResolvedValueOnce(mockAvailableStatus);
    vi.mocked(api.predictGraphRisk).mockResolvedValueOnce(mockPredictResponse);

    render(
      <GraphRiskWidget
        equipmentId="P-101A"
        deterministicRisk={0.42}
        currentVibration={5.42}
        currentTemp={84.6}
        currentFreq={236.4}
      />
    );

    await waitFor(() => {
      expect(screen.getByText("Available")).toBeInTheDocument();
    });

    // Check evaluate button is active
    const evalButton = screen.getByRole("button", { name: /Evaluate Graph Risk/i });
    expect(evalButton).toBeInTheDocument();

    fireEvent.click(evalButton);

    await waitFor(() => {
      expect(api.predictGraphRisk).toHaveBeenCalledTimes(1);
    });

    // Verify payload sent
    const calledWith = vi.mocked(api.predictGraphRisk).mock.calls[0][0];
    expect(calledWith.equipment_id).toBe("P-101A");
    expect(calledWith.features).toHaveLength(7);
    expect(calledWith.explain).toBe(true);
    expect(calledWith.deterministic_risk).toBe(0.42);

    // Verify rendered score and attribution
    await waitFor(() => {
      expect(screen.getByText("74.2%")).toBeInTheDocument();
      expect(screen.getByTestId("explanation-panel")).toBeInTheDocument();
      expect(screen.getByText("Predecessor Influence Attribution (2-Hop Subgraph)")).toBeInTheDocument();
      expect(screen.getByText("OBS-003")).toBeInTheDocument();
      expect(screen.getByText("62.0%")).toBeInTheDocument();
      expect(screen.getByText("OBS-002")).toBeInTheDocument();
      expect(screen.getByText("28.0%")).toBeInTheDocument();
      expect(screen.getByText(/Attribution is bounded by causal 2-hop backward horizon/)).toBeInTheDocument();
    });
  });

  it("handles explain checkbox toggling", async () => {
    vi.mocked(api.getGraphRiskStatus).mockResolvedValueOnce(mockAvailableStatus);
    vi.mocked(api.predictGraphRisk).mockResolvedValueOnce({
      ...mockPredictResponse,
      explanation: undefined,
    });

    render(<GraphRiskWidget equipmentId="P-101A" />);

    await waitFor(() => {
      expect(screen.getByText("Available")).toBeInTheDocument();
    });

    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toBeChecked();
    fireEvent.click(checkbox);
    expect(checkbox).not.toBeChecked();

    const evalButton = screen.getByRole("button", { name: /Evaluate Graph Risk/i });
    fireEvent.click(evalButton);

    await waitFor(() => {
      expect(api.predictGraphRisk).toHaveBeenCalledWith(
        expect.objectContaining({
          explain: false,
        })
      );
    });
  });

  it("displays error message gracefully if status or predict call fails", async () => {
    vi.mocked(api.getGraphRiskStatus).mockRejectedValueOnce(new Error("Network timeout"));

    render(<GraphRiskWidget equipmentId="P-101A" />);

    await waitFor(() => {
      expect(screen.getByText("Network timeout")).toBeInTheDocument();
      expect(screen.getByText("Disabled")).toBeInTheDocument();
    });
  });
});
