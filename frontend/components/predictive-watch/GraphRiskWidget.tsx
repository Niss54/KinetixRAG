"use client";

import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  HelpCircle,
  Info,
  Layers,
  Network,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";
import {
  getGraphRiskStatus,
  GraphRiskPredictRequest,
  GraphRiskPredictResponse,
  GraphRiskStatusResponse,
  predictGraphRisk,
} from "@/lib/api";

interface GraphRiskWidgetProps {
  equipmentId?: string;
  deterministicRisk?: number;
  currentVibration?: number;
  currentTemp?: number;
  currentFreq?: number;
}

export default function GraphRiskWidget({
  equipmentId = "P-101A",
  deterministicRisk = 0.42,
  currentVibration = 5.42,
  currentTemp = 84.6,
  currentFreq = 236.4,
}: GraphRiskWidgetProps) {
  const [statusData, setStatusData] = useState<GraphRiskStatusResponse | null>(null);
  const [prediction, setPrediction] = useState<GraphRiskPredictResponse | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<boolean>(true);
  const [scoring, setScoring] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [includeExplain, setIncludeExplain] = useState<boolean>(true);

  const fetchStatus = async () => {
    setLoadingStatus(true);
    setError(null);
    try {
      const data = await getGraphRiskStatus();
      setStatusData(data);
    } catch (err: any) {
      setError(err?.message || "Failed to connect to Graph Risk service.");
      // Graceful fallback status
      setStatusData({
        status: "disabled",
        model_available: false,
        schema_version: "1.0.0",
        feature_flag_enabled: false,
        feature_names: [],
        feature_dim: 7,
        detail: "Service unreachable; defaulted to disabled mode.",
      });
    } finally {
      setLoadingStatus(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handleEvaluate = async () => {
    setScoring(true);
    setError(null);
    try {
      // 7 canonical features:
      // [vibration_rms, vibration_peak, crest_factor, kurtosis, temp, temp_rate, dom_freq]
      const vibPeak = currentVibration * 1.6;
      const crest = currentVibration > 0 ? vibPeak / currentVibration : 1.4;
      const features = [
        Number(currentVibration.toFixed(2)),
        Number(vibPeak.toFixed(2)),
        Number(crest.toFixed(2)),
        4.2, // Kurtosis
        Number(currentTemp.toFixed(1)),
        0.85, // Rate of change (C/hr)
        Number(currentFreq.toFixed(1)),
      ];

      const req: GraphRiskPredictRequest = {
        sample_id: `SAMPLE-${Date.now().toString().slice(-4)}`,
        equipment_id: equipmentId,
        features,
        explain: includeExplain,
        deterministic_risk: deterministicRisk,
      };

      const res = await predictGraphRisk(req);
      setPrediction(res);
    } catch (err: any) {
      setError(err?.message || "Inference scoring error.");
    } finally {
      setScoring(false);
    }
  };

  const isAvailable = statusData?.model_available && statusData?.status === "available";

  return (
    <Card className="border border-purple-500/20 bg-card/60 backdrop-blur-sm" data-testid="graph-risk-widget">
      <CardHeader className="border-b py-3 px-4 flex flex-row items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-purple-500/10 text-purple-400">
            <BrainCircuit className="size-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <CardTitle className="text-sm font-semibold tracking-tight">
                GraphSAGE Risk Intelligence
              </CardTitle>
              <Badge variant="outline" className="text-[10px] font-mono border-purple-500/30 text-purple-400">
                Schema v{statusData?.schema_version || "1.0.0"}
              </Badge>
            </div>
            <CardDescription className="text-xs text-muted-foreground mt-0.5">
              Inductive GNN neighborhood attribution for rotating machinery
            </CardDescription>
          </div>
        </div>

        <CardAction className="flex items-center gap-2">
          {loadingStatus ? (
            <Spinner className="size-3 text-muted-foreground" />
          ) : isAvailable ? (
            <Badge variant="success" className="text-[11px] gap-1 py-0.5 px-2">
              <CheckCircle2 className="size-3" />
              <span>Available</span>
            </Badge>
          ) : (
            <Badge variant="outline" className="text-[11px] font-mono border-amber-500/40 text-amber-400 gap-1 py-0.5 px-2">
              <AlertTriangle className="size-3" />
              <span>{statusData?.status === "disabled" ? "Disabled" : "Model Not Trained"}</span>
            </Badge>
          )}

          <Badge variant="secondary" className="text-[10px] uppercase font-mono tracking-wider bg-zinc-800 text-zinc-300">
            Advisory Only
          </Badge>
        </CardAction>
      </CardHeader>

      <CardContent className="p-4 flex flex-col gap-4 text-xs">
        {error && (
          <div className="p-2.5 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-[11px] flex items-center gap-2">
            <AlertTriangle className="size-3.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Status & Architecture Card */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="p-3 rounded-lg bg-muted/40 border border-border/60 flex flex-col justify-between">
            <span className="text-muted-foreground font-medium text-[11px] flex items-center gap-1.5">
              <Layers className="size-3 text-cyan-400" />
              <span>Deterministic Rule Score</span>
            </span>
            <div className="mt-1 flex items-baseline gap-1.5">
              <span className="text-xl font-mono font-bold text-foreground">
                {(deterministicRisk * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] text-muted-foreground">ISO 10816-3</span>
            </div>
            <span className="text-[10px] text-muted-foreground mt-1">
              Primary operational source
            </span>
          </div>

          <div className="p-3 rounded-lg bg-muted/40 border border-border/60 flex flex-col justify-between">
            <span className="text-muted-foreground font-medium text-[11px] flex items-center gap-1.5">
              <Network className="size-3 text-purple-400" />
              <span>GraphSAGE Advisory Score</span>
            </span>
            <div className="mt-1 flex items-baseline gap-1.5">
              {prediction?.advisory_risk_score !== null && prediction?.advisory_risk_score !== undefined ? (
                <span className="text-xl font-mono font-bold text-purple-400">
                  {(prediction.advisory_risk_score * 100).toFixed(1)}%
                </span>
              ) : (
                <span className="text-xl font-mono font-bold text-muted-foreground">
                  --
                </span>
              )}
              <span className="text-[10px] text-muted-foreground">Uncalibrated</span>
            </div>
            <span className="text-[10px] text-muted-foreground mt-1">
              {isAvailable ? "Inductive 2-hop mean" : "Feature disabled / offline"}
            </span>
          </div>

          <div className="p-3 rounded-lg bg-muted/40 border border-border/60 flex flex-col justify-between">
            <span className="text-muted-foreground font-medium text-[11px] flex items-center gap-1.5">
              <ShieldCheck className="size-3 text-emerald-400" />
              <span>Governance Barrier</span>
            </span>
            <div className="mt-1">
              <Badge variant="outline" className="text-[10px] font-mono border-emerald-500/30 text-emerald-400">
                Payment Interlock Active
              </Badge>
            </div>
            <span className="text-[10px] text-muted-foreground mt-1">
              ML score cannot authorize funds
            </span>
          </div>
        </div>

        {/* Informative Guidance Banner */}
        {!isAvailable && (
          <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-[11px] leading-relaxed flex items-start gap-2.5">
            <Info className="size-4 shrink-0 text-amber-400 mt-0.5" />
            <div>
              <span className="font-semibold text-amber-200">Optional ML Intelligence Layer:</span>
              <p className="mt-0.5 text-amber-300/90">
                GraphSAGE is lazy-loaded and disabled by default (<code>ENABLE_GRAPH_RISK_MODEL=false</code>) to protect deployment memory budgets on free-tier infrastructure. KinetixRAG operates safely using its verified deterministic ISO 10816 rules and GraphRAG context.
              </p>
            </div>
          </div>
        )}

        {/* Interactive Evaluation Action */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-border/40">
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-1.5 text-muted-foreground cursor-pointer text-xs">
              <input
                type="checkbox"
                checked={includeExplain}
                onChange={(e) => setIncludeExplain(e.target.checked)}
                className="rounded border-zinc-700 bg-zinc-900 text-purple-600 focus:ring-purple-500"
              />
              <span>Compute bounded 2-hop graph explainability</span>
            </label>
          </div>

          <Button
            size="sm"
            onClick={handleEvaluate}
            disabled={scoring}
            className="text-xs bg-purple-600 hover:bg-purple-500 text-white gap-1.5"
          >
            {scoring ? <Spinner className="size-3" /> : <RefreshCw className="size-3" />}
            <span>Evaluate Graph Risk</span>
          </Button>
        </div>

        {/* Explainability Panel */}
        {prediction?.explanation && (
          <div className="p-3 rounded-lg bg-purple-950/20 border border-purple-500/30 flex flex-col gap-2.5" data-testid="explanation-panel">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-foreground text-xs flex items-center gap-1.5">
                <Network className="size-3.5 text-purple-400" />
                <span>Predecessor Influence Attribution ({prediction.explanation.hops_analyzed}-Hop Subgraph)</span>
              </span>
              <Badge variant="outline" className="text-[10px] font-mono border-purple-500/40 text-purple-400">
                Status: {prediction.explanation.status}
              </Badge>
            </div>

            {prediction.explanation.influential_nodes && prediction.explanation.influential_nodes.length > 0 ? (
              <div className="flex flex-col gap-1.5">
                {prediction.explanation.influential_nodes.map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2 rounded bg-background/60 border border-border/40 text-[11px]">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-semibold text-purple-300">{item.node_id}</span>
                      <span className="text-muted-foreground text-[10px]">({item.dominant_feature})</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-20 h-1.5 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-purple-500 rounded-full"
                          style={{ width: `${Math.round(item.influence_score * 100)}%` }}
                        />
                      </div>
                      <span className="font-mono text-xs font-medium text-foreground">
                        {(item.influence_score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-[11px] text-muted-foreground italic">
                No external predecessors detected in causal neighborhood (isolated observation).
              </p>
            )}

            <p className="text-[10px] text-muted-foreground italic mt-0.5 leading-normal">
              *{prediction.explanation.caveat}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
