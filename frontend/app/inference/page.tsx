"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import CodePanel from "../../components/CodePanel";
import Heatmap from "../../components/Heatmap";
import LineChart from "../../components/LineChart";
import RangeSlider from "../../components/RangeSlider";
import StatCard from "../../components/StatCard";
import { Checkpoint, CodeSnippet, fetchJson } from "../../lib/api";
import { formatCheckpointTimestamp } from "../../lib/time";
import MarkdownBlock from "../../components/MarkdownBlock";
import { inferenceEquations } from "../../lib/equations";

type SessionInfo = {
  session_id: string;
  tokenizer_type: string;
  param_count_m: number;
  cfg: Record<string, number | string>;
};

type DiagnosticsMeta = {
  diagnostic_id: string;
  token_ids: number[];
  token_labels: string[];
};

export default function InferencePage() {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [prompt, setPrompt] = useState("First Citizen:");
  const [maxNewTokens, setMaxNewTokens] = useState(200);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState<number | "">("");
  const [topP, setTopP] = useState(0.9);
  const [generatedText, setGeneratedText] = useState("");
  const [diagnostics, setDiagnostics] = useState<DiagnosticsMeta | null>(null);
  const [attentionMatrix, setAttentionMatrix] = useState<number[][]>([]);
  const [layerNorms, setLayerNorms] = useState<Array<{ layer: number; avg_norm: number }>>([]);
  const [logitLens, setLogitLens] = useState<Array<{ layer: number; predictions: { rank: number; token: string; prob: number }[] }>>([]);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [selectedPosition, setSelectedPosition] = useState(0);
  const [snippets, setSnippets] = useState<CodeSnippet[]>([]);
  const [snippetsLoading, setSnippetsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDiagnosticsLoading, setIsDiagnosticsLoading] = useState(false);

  const generatedCount = Math.max(0, generatedText.length - prompt.length);

  useEffect(() => {
    fetchJson<{ checkpoints: Checkpoint[] }>("/api/checkpoints")
      .then((data) => setCheckpoints(data.checkpoints))
      .catch((err) => setError((err as Error).message));
  }, []);

  const sortedCheckpoints = useMemo(
    () => [...checkpoints].sort((a, b) => b.mtime - a.mtime),
    [checkpoints]
  );

  useEffect(() => {
    if (selectedCheckpoint || sortedCheckpoints.length === 0) {
      return;
    }
    setSelectedCheckpoint(sortedCheckpoints[0].id);
  }, [sortedCheckpoints, selectedCheckpoint]);

  const loadSession = async () => {
    setError(null);
    try {
      if (!selectedCheckpoint) {
        throw new Error("Select a checkpoint first.");
      }
      const data = await fetchJson<SessionInfo>("/api/inference/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint_id: selectedCheckpoint }),
      });
      setSession(data);
      setGeneratedText("");
      setDiagnostics(null);
      setAttentionMatrix([]);
      setLayerNorms([]);
      setLogitLens([]);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const generate = async () => {
    if (!session) return;
    setError(null);
    try {
      const data = await fetchJson<{ generated_text: string }>(
        `/api/inference/sessions/${session.session_id}/generate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt,
            max_new_tokens: maxNewTokens,
            temperature,
            top_k: topK === "" ? null : topK,
            top_p: topP,
          }),
        }
      );
      setGeneratedText(data.generated_text);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const runDiagnostics = useCallback(async () => {
    if (!session) return;
    setError(null);
    setIsDiagnosticsLoading(true);
    try {
      const data = await fetchJson<DiagnosticsMeta>(
        `/api/inference/sessions/${session.session_id}/diagnostics`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt }),
        }
      );
      setDiagnostics(data);
      setSelectedLayer(0);
      setSelectedHead(0);
      setSelectedPosition(Math.max(0, data.token_ids.length - 2));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsDiagnosticsLoading(false);
    }
  }, [prompt, session]);

  useEffect(() => {
    if (!diagnostics) return;
    fetchJson<{ attention: number[][] }>(
      `/api/inference/diagnostics/${diagnostics.diagnostic_id}/attention?layer=${selectedLayer}&head=${selectedHead}`
    )
      .then((data) => setAttentionMatrix(data.attention))
      .catch((err) => setError((err as Error).message));
  }, [diagnostics, selectedLayer, selectedHead]);

  useEffect(() => {
    if (!diagnostics) return;
    fetchJson<{ layers: Array<{ layer: number; predictions: { rank: number; token: string; prob: number }[] }> }>(
      `/api/inference/diagnostics/${diagnostics.diagnostic_id}/logit-lens?position=${selectedPosition}`
    )
      .then((data) => setLogitLens(data.layers))
      .catch((err) => setError((err as Error).message));
  }, [diagnostics, selectedPosition]);

  useEffect(() => {
    if (!diagnostics) return;
    fetchJson<{ layers: Array<{ layer: number; avg_norm: number }> }>(
      `/api/inference/diagnostics/${diagnostics.diagnostic_id}/layer-norms`
    )
      .then((data) => setLayerNorms(data.layers))
      .catch((err) => setError((err as Error).message));
  }, [diagnostics]);

  const loadSnippets = async () => {
    setError(null);
    try {
      const data = await fetchJson<{ snippets: CodeSnippet[] }>("/api/docs/inference-code");
      setSnippets(data.snippets);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const layersCount = session ? Number(session.cfg?.n_layers || 0) : 0;
  const headsCount = session ? Number(session.cfg?.n_heads || 0) : 0;
  const maxLayerIndex = Math.max(0, layersCount - 1);
  const maxHeadIndex = Math.max(0, headsCount - 1);

  useEffect(() => {
    if (!session) {
      return;
    }
    const timeout = setTimeout(() => {
      runDiagnostics();
    }, 400);
    return () => clearTimeout(timeout);
  }, [prompt, runDiagnostics, session]);

  useEffect(() => {
    let active = true;
    setSnippetsLoading(true);
    const timeout = setTimeout(() => {
      loadSnippets().finally(() => {
        if (active) {
          setSnippetsLoading(false);
        }
      });
    }, 400);
    return () => {
      active = false;
      clearTimeout(timeout);
    };
  }, []);

  return (
    <>
      <section className="section">
        <div className="section-title">
          <h2>Select Model</h2>
          <p>Load a checkpoint for inference.</p>
        </div>
        <div className="card">
          <div className="inline-row" style={{ alignItems: "center" }}>
            <select value={selectedCheckpoint} onChange={(event) => setSelectedCheckpoint(event.target.value)}>
              <option value="">Select checkpoint</option>
              {sortedCheckpoints.map((ckpt) => (
                <option key={ckpt.id} value={ckpt.id}>
                  {formatCheckpointTimestamp(new Date(ckpt.mtime * 1000))} · {ckpt.name}
                </option>
              ))}
            </select>
            <button className="primary" onClick={loadSession}>Load Model</button>
          </div>

          {session && (
            <div className="grid-3" style={{ marginTop: 16 }}>
              <StatCard label="Tokenizer" value={session.tokenizer_type} />
              <StatCard label="Parameters" value={`${session.param_count_m}M`} />
              <StatCard label="d_model" value={session.cfg?.d_model || "-"} />
              <StatCard label="n_layers" value={session.cfg?.n_layers || "-"} />
              <StatCard label="n_heads" value={session.cfg?.n_heads || "-"} />
              <StatCard label="n_ctx" value={session.cfg?.n_ctx || "-"} />
            </div>
          )}
          {session && (
            <div style={{ marginTop: 16 }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Config</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(session.cfg || {}).map(([key, value]) => (
                    <tr key={key}>
                      <td>{key}</td>
                      <td>{String(value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Generation Settings</h2>
          <p>Sampling controls for text generation.</p>
        </div>
        <div className="card">
          <div className="grid-2">
            <div>
              <label>Prompt</label>
              <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
            </div>
            <div className="grid-3">
              <div>
                <label>Max New Tokens</label>
                <input type="number" value={maxNewTokens} onChange={(event) => setMaxNewTokens(Number(event.target.value))} />
              </div>
              <div>
                <label>Temperature</label>
                <input type="number" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
              </div>
              <div>
                <label>Top-k</label>
                <input
                  type="number"
                  value={topK}
                  onChange={(event) => setTopK(event.target.value === "" ? "" : Number(event.target.value))}
                />
              </div>
              <div>
                <label>Top-p</label>
                <input type="number" step="0.05" value={topP} onChange={(event) => setTopP(Number(event.target.value))} />
              </div>
            </div>
          </div>
          <div className="inline-row" style={{ marginTop: 12 }}>
            <button className="primary" onClick={generate} disabled={!session}>Generate</button>
          </div>
          {error && <p style={{ color: "#b42318" }}>{error}</p>}
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Understand</h2>
          <p>Sampling equations and references.</p>
        </div>
        <div className="card">
          <details className="expander">
            <summary>Equations</summary>
            <div className="expander-content">
              <MarkdownBlock content={inferenceEquations} />
            </div>
          </details>
          <details className="expander">
            <summary>Code Snippets</summary>
            <div className="expander-content">
              {snippetsLoading ? (
                <p>Loading code snippets...</p>
              ) : snippets.length === 0 ? (
                <p>No snippets available yet.</p>
              ) : (
                snippets.map((snippet) => <CodePanel key={snippet.title} snippet={snippet} />)
              )}
            </div>
          </details>
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Generated Text</h2>
          <p>Model output based on your prompt.</p>
        </div>
        <div className="card">
          <textarea value={generatedText} readOnly style={{ minHeight: 220 }} />
          {generatedText && (
            <p style={{ marginTop: 8 }}>
              Prompt length {prompt.length} chars • Generated {generatedCount} chars
            </p>
          )}
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Model Internals</h2>
          <p>Attention, logit lens, and layer norms.</p>
        </div>
        <div className="card">
          {diagnostics ? (
            <>
              <div className="grid-3">
                <RangeSlider
                  label="Layer"
                  min={0}
                  max={maxLayerIndex}
                  value={selectedLayer}
                  onChange={setSelectedLayer}
                  disabled={maxLayerIndex === 0}
                />
                <RangeSlider
                  label="Head"
                  min={0}
                  max={maxHeadIndex}
                  value={selectedHead}
                  onChange={setSelectedHead}
                  disabled={maxHeadIndex === 0}
                />
                <div>
                  <label>Logit Lens Position</label>
                  <input
                    type="number"
                    min={0}
                    max={Math.max(0, diagnostics.token_ids.length - 1)}
                    value={selectedPosition}
                    onChange={(event) => setSelectedPosition(Number(event.target.value))}
                  />
                </div>
              </div>

              <div style={{ marginTop: 16 }}>
                <h3>Attention Heatmap</h3>
                <Heatmap matrix={attentionMatrix} labels={diagnostics.token_labels} />
              </div>

              <div style={{ marginTop: 24 }}>
                <h3>Logit Lens</h3>
                <div className="card" style={{ boxShadow: "none", background: "var(--card-muted)" }}>
                  {logitLens.length === 0 ? (
                    <p>Loading logit lens...</p>
                  ) : (
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Layer</th>
                          <th>Top Predictions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {logitLens.map((row) => (
                          <tr key={row.layer}>
                            <td>{row.layer}</td>
                            <td>
                              {row.predictions.map((pred) => (
                                <span key={`${row.layer}-${pred.rank}`} className="badge" style={{ marginRight: 8 }}>
                                  {pred.rank}. {pred.token} ({(pred.prob * 100).toFixed(1)}%)
                                </span>
                              ))}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>

              <div style={{ marginTop: 24 }}>
                <h3>Layer Norms</h3>
                <LineChart
                  data={layerNorms.map((row) => ({ layer: row.layer, norm: row.avg_norm }))}
                  xKey="layer"
                  lines={[{ dataKey: "norm", name: "Avg Norm", color: "var(--accent)" }]}
                />
              </div>
            </>
          ) : (
            <p>
              {session
                ? isDiagnosticsLoading
                  ? "Running diagnostics..."
                  : "Diagnostics update automatically when you change the prompt."
                : "Load a checkpoint to view diagnostics."}
            </p>
          )}
        </div>
      </section>
    </>
  );
}
