import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Database,
  Gauge,
  ImagePlus,
  Layers3,
  Play,
  ShieldCheck,
  Sparkles,
  Upload,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';
const FALLBACK_CLASSES = ['W180', 'W210', 'W300', 'W500'];
const GALLERY_ACCENTS = ['#e98a33', '#6cb7ff', '#76df9d', '#f15f59', '#b58dff', '#ffc85c'];
const UNIFORM_PROBABILITY_SPREAD = 0.01;
const RANDOM_CONFIDENCE_TOLERANCE = 0.015;

const pipelineSteps = [
  { title: 'Image Import', icon: Upload },
  { title: 'Preprocessing', icon: Sparkles },
  { title: 'CNN Inference', icon: Layers3 },
  { title: 'Grade Mapping', icon: Gauge },
  { title: 'Result Logging', icon: ShieldCheck },
];

function formatPercent(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return '0.0%';
  return `${(numericValue * 100).toFixed(1)}%`;
}

function formatDate(date = new Date()) {
  return new Intl.DateTimeFormat('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

function inferBusinessBand(label) {
  const normalizedLabel = String(label || '').trim().toUpperCase();
  const gradeBands = {
    W180: 'Premium Large Kernel',
    W210: 'Jumbo Kernel',
    W300: 'Standard Kernel',
    W500: 'Small Kernel',
  };
  return gradeBands[normalizedLabel] || 'Custom Grade';
}

function diagnosePrediction(prediction, probabilityValues) {
  if (!prediction || !probabilityValues.length) return null;
  const maxP = Math.max(...probabilityValues);
  const minP = Math.min(...probabilityValues);
  const randomConfidence = 1 / probabilityValues.length;
  const modelConfidence = Number(prediction.confidence) || maxP;
  const spread = maxP - minP;
  if (
    spread < UNIFORM_PROBABILITY_SPREAD &&
    Math.abs(modelConfidence - randomConfidence) < RANDOM_CONFIDENCE_TOLERANCE
  ) {
    return {
      type: 'uniform',
      message:
        'Model returned near-equal scores for all classes (~' +
        formatPercent(randomConfidence) +
        ' each). This usually means the model is not trained for the current W-grade dataset yet, ' +
        'or the loaded model file does not match the current labels.',
    };
  }
  if (modelConfidence < 0.4) {
    return {
      type: 'low',
      message:
        'Low confidence (' +
        formatPercent(modelConfidence) +
        '). The model is unsure. Add more training images or retrain longer for grade ' +
        prediction.predicted_grade + '.',
    };
  }
  return null;
}

function App() {
  const [health, setHealth] = useState({ status: 'checking', model_loaded: false, classes: FALLBACK_CLASSES });
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionLog, setSessionLog] = useState([]);

  useEffect(() => {
    let isMounted = true;
    async function fetchHealth() {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) throw new Error('Failed to reach backend.');
        const data = await response.json();
        if (isMounted) setHealth(data);
      } catch (fetchError) {
        if (isMounted) {
          setHealth({ status: 'offline', model_loaded: false, classes: FALLBACK_CLASSES });
          setError(fetchError.message);
        }
      }
    }
    fetchHealth();
    return () => { isMounted = false; };
  }, []);

  useEffect(() => {
    if (!selectedFile) { setPreviewUrl(''); return undefined; }
    if (typeof URL.createObjectURL !== 'function') { setPreviewUrl(''); return undefined; }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => { if (typeof URL.revokeObjectURL === 'function') URL.revokeObjectURL(objectUrl); };
  }, [selectedFile]);

  const activeClasses = health.classes?.length ? health.classes : FALLBACK_CLASSES;

  const probabilityEntries = useMemo(() => {
    if (!prediction?.class_probabilities) {
      return activeClasses.slice(0, 8).map((label, index) => ({
        name: label.toUpperCase(),
        value: 0,
        fill: GALLERY_ACCENTS[index % GALLERY_ACCENTS.length],
      }));
    }
    return Object.entries(prediction.class_probabilities)
      .map(([name, value], index) => ({
        name: name.toUpperCase(),
        value: Number(value) || 0,
        fill: GALLERY_ACCENTS[index % GALLERY_ACCENTS.length],
      }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8);
  }, [activeClasses, prediction]);

  const probabilityValues = useMemo(() => {
    if (!prediction?.class_probabilities) return [];
    return Object.values(prediction.class_probabilities).map(Number).filter(Number.isFinite);
  }, [prediction]);

  const diagnosis = useMemo(
    () => diagnosePrediction(prediction, probabilityValues),
    [prediction, probabilityValues]
  );

  const isUniformPrediction = diagnosis?.type === 'uniform';

  const pieEntries = useMemo(
    () => probabilityEntries.slice(0, 5).map((entry) => ({ ...entry })),
    [probabilityEntries]
  );

  const galleryCards = useMemo(() => {
    if (!prediction?.class_probabilities) {
      return activeClasses.slice(0, 8).map((label, index) => ({
        id: `${label}-${index}`,
        label,
        score: '—',
        accent: GALLERY_ACCENTS[index % GALLERY_ACCENTS.length],
        isWinner: false,
        rawValue: 0,
      }));
    }
    const winnerName = prediction.predicted_grade?.toUpperCase();
    return probabilityEntries.map((entry, index) => ({
      id: `${entry.name}-${index}`,
      label: entry.name.toLowerCase(),
      score: formatPercent(entry.value),
      accent: entry.fill,
      isWinner: entry.name === winnerName,
      rawValue: entry.value,
    }));
  }, [activeClasses, previewUrl, prediction, probabilityEntries]);

  const metrics = useMemo(() => {
    const confidence = prediction?.confidence ?? 0;
    const consistency = prediction ? Math.min(Math.max(confidence * 0.92, 0), 1) : 0;
    const readiness = health.status === 'ok' && health.model_loaded ? 1 : 0.2;
    return [
      { label: 'Classification Confidence', value: confidence, tone: 'warm' },
      { label: 'Model Readiness', value: readiness, tone: 'cool' },
      { label: 'Consistency Estimate', value: consistency, tone: 'neutral' },
    ];
  }, [health, prediction]);

  async function handlePredict() {
    if (!selectedFile) { setError('Choose an image first so we can run grading.'); return; }
    if (health.status !== 'ok') { setError('Backend is offline. Start FastAPI first, then try prediction again.'); return; }
    if (!health.model_loaded) { setError('Model weights are missing. Train the model or add cashew_model.h5 before prediction.'); return; }
    setLoading(true);
    setError('');
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
        cache: 'no-store',
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Prediction failed.');
      setPrediction(data);
      setSessionLog((prev) =>
        [
          {
            id: Date.now(),
            fileName: selectedFile.name,
            predictedGrade: data.predicted_grade,
            confidence: data.confidence,
            businessBand: inferBusinessBand(data.predicted_grade),
            processedAt: formatDate(),
          },
          ...prev,
        ].slice(0, 6)
      );
    } catch (predictError) {
      setError(predictError.message);
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    setPrediction(null);
    setError('');
    event.target.value = '';
  }

  return (
    <div className="app-shell">
      <div className="app-backdrop" />

      {/* ── Top nav ── */}
      <header className="hero-panel">
        <div className="hero-brand">
          <img src="/logo512.jpg" alt="Cashew App Logo" className="hero-emblem" />
          <h1>Cashew Quality Analytics and Grading System</h1>
        </div>
        <div className="hero-copy">
          <p className="hero-subtitle">Adapted for the updated W-grade dataset: W180, W210, W300, and W500.</p>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="dashboard">
        <section className="dashboard-grid">

          {/* ── Sidebar: col 1, rows 1-3 ── */}
          <aside className="panel panel-sidebar">
            <div className="panel-heading">
              <span>Dataset Management</span>
              <Activity size={14} />
            </div>

            <label className="action-button action-primary">
              <input type="file" accept="image/*" onChange={handleFileChange} hidden />
              <ImagePlus size={15} />
              <span>Load Kernel Image</span>
            </label>

            <button className="action-button action-secondary" type="button" onClick={handlePredict} disabled={loading}>
              <Play size={15} />
              <span>{loading ? 'Running…' : 'Start Grading Run'}</span>
            </button>

            <a className="action-button action-ghost" href={`${API_BASE_URL}/docs`} target="_blank" rel="noreferrer">
              <Database size={15} />
              <span>Open API Docs</span>
            </a>

            <div className="status-stack">
              <div className="stat-card">
                <span className="stat-label">Backend</span>
                <strong className={`stat-value ${health.status === 'ok' ? 'good' : 'warn'}`}>
                  {health.status === 'ok' ? 'Connected' : 'Offline'}
                </strong>
              </div>
              <div className="stat-card">
                <span className="stat-label">Model</span>
                <strong className={`stat-value ${health.model_loaded ? 'good' : 'warn'}`}>
                  {health.model_loaded ? 'Loaded' : 'Missing'}
                </strong>
              </div>
              <div className="stat-card">
                <span className="stat-label">Classes</span>
                <strong className="stat-value">{activeClasses.length}</strong>
              </div>
              <div className="stat-card">
                <span className="stat-label">File</span>
                <strong className="stat-value stat-file">{selectedFile?.name || 'None'}</strong>
              </div>
            </div>

            <div className="class-cloud">
              {activeClasses.map((label) => (
                <span
                  key={label}
                  className={`class-chip ${
                    prediction?.predicted_grade?.toLowerCase() === label.toLowerCase() ? 'class-chip-winner' : ''
                  }`}
                >
                  {label.toUpperCase()}
                </span>
              ))}
            </div>

            {/* Top Grade Clusters */}
            <div className="panel-defects panel-sidebar-inset">
              <div className="panel-heading">
                <span>Top Grade Clusters</span>
                <span className="panel-mini">Session Snapshot</span>
              </div>
              <div className="defect-list">
                {probabilityEntries.slice(0, 4).map((entry) => (
                  <div key={entry.name} className="defect-card">
                    <div className="defect-thumb" style={{ borderColor: entry.fill }}>
                      {previewUrl ? <img src={previewUrl} alt={entry.name} /> : <span>{entry.name}</span>}
                    </div>
                    <div>
                      <strong>{entry.name}</strong>
                      <p>{inferBusinessBand(entry.name.toLowerCase())}</p>
                      <small style={{ color: entry.fill }}>{formatPercent(entry.value)}</small>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </aside>

          {/* ── Gallery: col 2, row 1 ── */}
          <section className="panel panel-gallery">
            <div className="panel-heading">
              <span>Dataset Gallery &amp; Annotation</span>
              <span className="panel-mini">Live Prediction Workspace</span>
            </div>

            {/* Flex Layout for Side-by-Side Gallery */}
            <div className="gallery-layout">
              {/* Left Column: Small Cards */}
              <div className="gallery-tiles-column">
                {galleryCards.slice(0, 4).map((card, index) => (
                  <div
                    key={card.id}
                    className={`gallery-tile ${card.isWinner ? 'gallery-tile-winner' : ''}`}
                  >
                    <div className="gallery-placeholder">{card.label.toUpperCase()}</div>
                    <span className="tile-tag" style={{ backgroundColor: card.accent }}>{card.label}</span>
                    <span className="tile-score" style={card.isWinner ? { color: '#3fb950', fontWeight: 700 } : {}}>
                      {card.score}
                    </span>
                    {card.isWinner && <span className="tile-winner-badge">✓ BEST</span>}
                  </div>
                ))}
              </div>

              {/* Right Area: Large Spotlight Image */}
              <div className={`spotlight-card ${isUniformPrediction ? 'spotlight-warn' : ''}`}>
                <div className="spotlight-image-wrap">
                  {previewUrl
                    ? <img src={previewUrl} alt="Selected preview" className="spotlight-image" />
                    : <div className="spotlight-empty">Upload a kernel image to inspect it here.</div>
                  }
                </div>
                <div className="spotlight-caption">
                  <span>Predicted Grade</span>
                  <strong
                    data-testid="predicted-grade"
                    style={
                      prediction && !isUniformPrediction
                        ? { color: '#3fb950' }
                        : isUniformPrediction
                        ? { color: '#f85149' }
                        : {}
                    }
                  >
                    {prediction?.predicted_grade || 'Awaiting inference'}
                  </strong>
                  <small>
                    {prediction
                      ? `${formatPercent(prediction.confidence)} confidence`
                      : 'Confidence and ranking will appear here.'}
                  </small>
                  {diagnosis && (
                    <div className="error-banner diagnosis-banner" style={{ marginTop: 8 }}>
                      <AlertTriangle size={13} style={{ flexShrink: 0 }} />
                      <span>{diagnosis.message}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </section>

          {/* ── Analytics: col 3, row 1 ── */}
          <section className="panel panel-analytics">
            <div className="panel-heading">
              <span>Grading Statistics</span>
              <CheckCircle2 size={14} />
            </div>

            <div className="analytics-summary">
              <div>
                <span className="stat-label">Current Grade Band</span>
                <strong className="summary-grade" data-testid="business-band">
                  {prediction ? inferBusinessBand(prediction.predicted_grade) : 'Not graded yet'}
                </strong>
              </div>
              <div className="summary-count">
                <span className="stat-label">Tracked Labels</span>
                <strong>{activeClasses.length}</strong>
              </div>
            </div>

            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={probabilityEntries} margin={{ top: 4, right: 4, left: -16, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: '#484f5a', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis
                    tickFormatter={(v) => `${Math.round(v * 100)}%`}
                    tick={{ fill: '#484f5a', fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    formatter={(v) => formatPercent(v)}
                    contentStyle={{ background: '#161b22', border: '1px solid #2a3142', borderRadius: 8, fontSize: 12 }}
                    labelStyle={{ color: '#e6edf3' }}
                    itemStyle={{ color: '#8b949e' }}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {probabilityEntries.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-wrap pie-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={pieEntries} innerRadius={50} outerRadius={62} paddingAngle={3} dataKey="value" stroke="none">
                    {pieEntries.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} stroke="none" />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(v) => formatPercent(v)}
                    contentStyle={{ background: '#161b22', border: '1px solid #2a3142', borderRadius: 8, fontSize: 12 }}
                    itemStyle={{ color: '#8b949e' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="legend-list">
              {pieEntries.map((entry) => (
                <div key={entry.name} className="legend-item">
                  <span className="legend-dot" style={{ backgroundColor: entry.fill }} />
                  <span>{entry.name}</span>
                  <strong>{formatPercent(entry.value)}</strong>
                </div>
              ))}
            </div>
          </section>

          {/* ── Pipeline: col 2, row 2 ── */}
          <section className="panel panel-pipeline">
            <div className="panel-heading">
              <span>Processing Pipeline</span>
              <span className="panel-mini">Inference Steps</span>
            </div>
            <div className="pipeline-row">
              {pipelineSteps.map((step, index) => {
                const Icon = step.icon;
                return (
                  <React.Fragment key={step.title}>
                    <div className="pipeline-step">
                      <div className="pipeline-icon"><Icon size={18} /></div>
                      <span>{step.title}</span>
                    </div>
                    {index < pipelineSteps.length - 1 && <ArrowRight className="pipeline-arrow" size={14} />}
                  </React.Fragment>
                );
              })}
            </div>
          </section>

          {/* ── Confidence: col 3, row 2 ── */}
          <section className="panel panel-confidence">
            <div className="panel-heading">
              <span>Confidence &amp; Consistency</span>
              <span className="panel-mini">Live Readout</span>
            </div>
            <div className="meter-stack">
              {metrics.map((metric) => (
                <div key={metric.label} className="meter-block">
                  <div className="meter-label-row">
                    <span>{metric.label}</span>
                    <strong>{formatPercent(metric.value)}</strong>
                  </div>
                  <div className="meter-track">
                    <div className={`meter-fill meter-${metric.tone}`} style={{ width: `${metric.value * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
            {diagnosis && (
              <div className="error-banner diagnosis-banner" style={{ marginTop: '10px', marginBottom: '12px' }}>
                <AlertTriangle size={13} style={{ flexShrink: 0 }} />
                <div>
                  <strong>Model Diagnosis</strong>
                  <p style={{ margin: '3px 0 0', fontSize: 11, lineHeight: 1.5 }}>{diagnosis.message}</p>
                  <ul style={{ margin: '4px 0 0 14px', fontSize: 10, lineHeight: 1.6 }}>
                    <li>Verify <code>class_probabilities</code> are not all equal.</li>
                    <li>Test <code>/predict</code> in <a href={`${API_BASE_URL}/docs`} target="_blank" rel="noreferrer">API Docs</a>.</li>
                    <li>Ensure model trained on W180, W210, W300, W500.</li>
                    <li>Confirm preprocessing matches training.</li>
                  </ul>
                </div>
              </div>
            )}
            {error && <div className="error-banner" style={{ marginBottom: 12 }}>{error}</div>}
          </section>

          {/* ── Batch Log: col 2-3, row 3 ── */}
          <section className="panel panel-log">
            <div className="panel-heading">
              <span>Batch Processing Log</span>
              <span className="panel-mini">Latest Requests</span>
            </div>
            <div className="log-table">
              <div className="log-row log-head">
                <span>File</span>
                <span>Grade</span>
                <span>Band</span>
                <span>Confidence</span>
              </div>
              {(sessionLog.length
                ? sessionLog
                : [{ id: 'placeholder', fileName: 'waiting_for_input.jpg', predictedGrade: '?', businessBand: 'Pending', confidence: 0 }]
              ).map((entry) => (
                <div key={entry.id} className="log-row">
                  <span>{entry.fileName}</span>
                  <span>{entry.predictedGrade}</span>
                  <span>{entry.businessBand}</span>
                  <span
                    style={
                      Number(entry.confidence) > 0.6
                        ? { color: '#3fb950' }
                        : Number(entry.confidence) > 0.3
                        ? { color: '#d29922' }
                        : { color: '#f85149' }
                    }
                  >
                    {formatPercent(entry.confidence)}
                  </span>
                </div>
              ))}
            </div>
          </section>

        </section>
      </main>
    </div>
  );
}

export default App;