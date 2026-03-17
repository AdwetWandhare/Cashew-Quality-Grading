import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
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
const FALLBACK_CLASSES = Array.from({ length: 22 }, (_, index) => `s${index}`);
const GALLERY_ACCENTS = ['#e98a33', '#6cb7ff', '#76df9d', '#f15f59', '#b58dff', '#ffc85c'];

const pipelineSteps = [
  { title: 'Image Import', icon: Upload },
  { title: 'Preprocessing', icon: Sparkles },
  { title: 'CNN Inference', icon: Layers3 },
  { title: 'Grade Mapping', icon: Gauge },
  { title: 'Result Logging', icon: ShieldCheck },
];

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
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
  const numericPart = Number(label.replace('s', ''));

  if (numericPart <= 6) {
    return 'Premium Range';
  }
  if (numericPart <= 14) {
    return 'Standard Range';
  }
  return 'Utility Range';
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
        if (!response.ok) {
          throw new Error('Failed to reach backend.');
        }

        const data = await response.json();
        if (isMounted) {
          setHealth(data);
        }
      } catch (fetchError) {
        if (isMounted) {
          setHealth({
            status: 'offline',
            model_loaded: false,
            classes: FALLBACK_CLASSES,
          });
          setError(fetchError.message);
        }
      }
    }

    fetchHealth();
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl('');
      return undefined;
    }

    if (typeof URL.createObjectURL !== 'function') {
      setPreviewUrl('');
      return undefined;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => {
      if (typeof URL.revokeObjectURL === 'function') {
        URL.revokeObjectURL(objectUrl);
      }
    };
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
        value,
        fill: GALLERY_ACCENTS[index % GALLERY_ACCENTS.length],
      }))
      .sort((left, right) => right.value - left.value)
      .slice(0, 8);
  }, [activeClasses, prediction]);

  const pieEntries = useMemo(
    () => probabilityEntries.slice(0, 5).map((entry) => ({ ...entry })),
    [probabilityEntries]
  );

  const galleryCards = useMemo(() => {
    if (!previewUrl) {
      return activeClasses.slice(0, 12).map((label, index) => ({
        id: `${label}-${index}`,
        label,
        score: '0.0%',
        accent: GALLERY_ACCENTS[index % GALLERY_ACCENTS.length],
      }));
    }

    return probabilityEntries.slice(0, 12).map((entry, index) => ({
      id: `${entry.name}-${index}`,
      label: entry.name.toLowerCase(),
      score: formatPercent(entry.value),
      accent: entry.fill,
    }));
  }, [activeClasses, previewUrl, probabilityEntries]);

  const metrics = useMemo(() => {
    const confidence = prediction?.confidence ?? 0;
    const consistency = health.model_loaded ? Math.max(confidence * 0.92, 0.18) : 0.12;
    const readiness = health.status === 'ok' ? 0.94 : 0.2;

    return [
      { label: 'Classification Confidence', value: confidence, tone: 'warm' },
      { label: 'Model Readiness', value: readiness, tone: 'cool' },
      { label: 'Consistency Estimate', value: consistency, tone: 'neutral' },
    ];
  }, [health, prediction]);

  async function handlePredict() {
    if (!selectedFile) {
      setError('Choose an image first so we can run grading.');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed.');
      }

      setPrediction(data);
      setSessionLog((previous) => [
        {
          id: Date.now(),
          fileName: selectedFile.name,
          predictedGrade: data.predicted_grade,
          confidence: data.confidence,
          businessBand: inferBusinessBand(data.predicted_grade),
          processedAt: formatDate(),
        },
        ...previous,
      ].slice(0, 6));
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
  }

  return (
    <div className="app-shell">
      <div className="app-backdrop" />
      <main className="dashboard">
        <header className="hero-panel">
          <div className="hero-brand">
            <div className="hero-emblem">C</div>
            <div>
              <p className="eyebrow">Deep Learning Inspection Console</p>
              <h1>Cashew Quality Analytics and Grading System</h1>
            </div>
          </div>
          <div className="hero-copy">
            <p className="hero-title">Dataset and Batch Processing Analytics</p>
            <p className="hero-subtitle">
              Inspired by industrial grading dashboards, adapted for your `s0` to `s21` grading workflow.
            </p>
          </div>
        </header>

        <section className="dashboard-grid">
          <aside className="panel panel-sidebar">
            <div className="panel-heading">
              <span>Dataset Management</span>
              <Activity size={16} />
            </div>

            <label className="action-button action-primary">
              <input type="file" accept="image/*" onChange={handleFileChange} hidden />
              <ImagePlus size={18} />
              <span>Load Kernel Image</span>
            </label>

            <button className="action-button action-secondary" type="button" onClick={handlePredict} disabled={loading}>
              <Play size={18} />
              <span>{loading ? 'Running Prediction...' : 'Start Grading Run'}</span>
            </button>

            <a className="action-button action-ghost" href={`${API_BASE_URL}/docs`} target="_blank" rel="noreferrer">
              <Database size={18} />
              <span>Open API Docs</span>
            </a>

            <div className="status-stack">
              <div className="stat-card">
                <span className="stat-label">Backend Status</span>
                <strong className={`stat-value ${health.status === 'ok' ? 'good' : 'warn'}`}>
                  {health.status === 'ok' ? 'Connected' : 'Offline'}
                </strong>
              </div>
              <div className="stat-card">
                <span className="stat-label">Model Weights</span>
                <strong className={`stat-value ${health.model_loaded ? 'good' : 'warn'}`}>
                  {health.model_loaded ? 'Loaded' : 'Missing'}
                </strong>
              </div>
              <div className="stat-card">
                <span className="stat-label">Active Grading Classes</span>
                <strong className="stat-value">{activeClasses.length}</strong>
              </div>
              <div className="stat-card">
                <span className="stat-label">Selected File</span>
                <strong className="stat-value stat-file">{selectedFile?.name || 'No image chosen'}</strong>
              </div>
            </div>

            <div className="class-cloud">
              {activeClasses.slice(0, 22).map((label) => (
                <span key={label} className="class-chip">
                  {label.toUpperCase()}
                </span>
              ))}
            </div>
          </aside>

          <section className="panel panel-gallery">
            <div className="panel-heading">
              <span>Dataset Gallery and Annotation</span>
              <span className="panel-mini">Live Prediction Workspace</span>
            </div>

            <div className="gallery-grid">
              {galleryCards.map((card, index) => (
                <div key={card.id} className={`gallery-tile ${index === 4 ? 'gallery-tile-focus' : ''}`}>
                  {previewUrl ? (
                    <img src={previewUrl} alt="Selected cashew kernel" />
                  ) : (
                    <div className="gallery-placeholder">{card.label.toUpperCase()}</div>
                  )}
                  <span className="tile-tag" style={{ backgroundColor: card.accent }}>
                    {card.label}
                  </span>
                  <span className="tile-score">{card.score}</span>
                </div>
              ))}

              <div className="spotlight-card">
                {previewUrl ? (
                  <img src={previewUrl} alt="Selected preview" className="spotlight-image" />
                ) : (
                  <div className="spotlight-empty">Upload a kernel image to inspect it here.</div>
                )}

                <div className="spotlight-caption">
                  <span>Predicted Grade</span>
                  <strong data-testid="predicted-grade">{prediction?.predicted_grade || 'Awaiting inference'}</strong>
                  <small>
                    {prediction ? `${formatPercent(prediction.confidence)} confidence` : 'Confidence and ranking will appear here.'}
                  </small>
                </div>
              </div>
            </div>
          </section>

          <section className="panel panel-analytics">
            <div className="panel-heading">
              <span>Grading Statistics</span>
              <CheckCircle2 size={16} />
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
                <BarChart data={probabilityEntries}>
                  <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: '#c6d1e6', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tickFormatter={(value) => `${Math.round(value * 100)}%`} tick={{ fill: '#91a0bb', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip formatter={(value) => formatPercent(value)} />
                  <Bar dataKey="value" radius={[8, 8, 0, 0]}>
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
                  <Pie data={pieEntries} innerRadius={44} outerRadius={76} paddingAngle={3} dataKey="value">
                    {pieEntries.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => formatPercent(value)} />
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
                      <div className="pipeline-icon">
                        <Icon size={22} />
                      </div>
                      <span>{step.title}</span>
                    </div>
                    {index < pipelineSteps.length - 1 && <ArrowRight className="pipeline-arrow" size={18} />}
                  </React.Fragment>
                );
              })}
            </div>
          </section>

          <section className="panel panel-defects">
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
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="panel panel-log">
            <div className="panel-heading">
              <span>Batch Processing Log</span>
              <span className="panel-mini">Latest Requests</span>
            </div>

            <div className="log-table">
              <div className="log-row log-head">
                <span>File</span>
                <span>Prediction</span>
                <span>Band</span>
                <span>Confidence</span>
              </div>
              {(sessionLog.length ? sessionLog : [
                {
                  id: 'placeholder',
                  fileName: 'waiting_for_input.jpg',
                  predictedGrade: 's?',
                  businessBand: 'Pending',
                  confidence: 0,
                },
              ]).map((entry) => (
                <div key={entry.id} className="log-row">
                  <span>{entry.fileName}</span>
                  <span>{entry.predictedGrade}</span>
                  <span>{entry.businessBand}</span>
                  <span>{formatPercent(entry.confidence)}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="panel panel-confidence">
            <div className="panel-heading">
              <span>Overall Confidence and Consistency</span>
              <span className="panel-mini">Live Model Readout</span>
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

            {error ? <div className="error-banner">{error}</div> : null}
          </section>
        </section>
      </main>
    </div>
  );
}

export default App;
