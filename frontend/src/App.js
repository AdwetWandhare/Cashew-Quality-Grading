import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Cpu,
  ExternalLink,
  Gauge,
  ImagePlus,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Upload,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  RadialBar,
  RadialBarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';
const FALLBACK_CLASSES = ['W180', 'W210', 'W300', 'W500'];
const GRADE_INFO = {
  W180: { band: 'Premium Large', description: 'Top-tier export grade with large uniform kernels.' },
  W210: { band: 'Jumbo', description: 'Large kernels suited for premium retail and gifting.' },
  W300: { band: 'Standard', description: 'Balanced size for mainstream confectionery and snack use.' },
  W500: { band: 'Small', description: 'Smaller kernels ideal for blends and processing applications.' },
};
const GRADE_COLORS = {
  W180: '#b86a1c',
  W210: '#d28a22',
  W300: '#d7a45c',
  W500: '#8f6b46',
};

function formatPercent(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return '0.0%';
  return `${(numericValue * 100).toFixed(1)}%`;
}

function formatTime(timestamp = Date.now()) {
  return new Intl.DateTimeFormat('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    day: '2-digit',
    month: 'short',
  }).format(new Date(timestamp));
}

function timeAgo(timestamp) {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

function inferBand(grade) {
  return GRADE_INFO[String(grade || '').toUpperCase()]?.band || 'Custom Grade';
}

function diagnosePrediction(prediction) {
  if (!prediction?.class_probabilities) return null;
  const values = Object.values(prediction.class_probabilities).map(Number).filter(Number.isFinite);
  if (!values.length) return null;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const randomConfidence = 1 / values.length;
  const confidence = Number(prediction.confidence) || max;

  if (max - min < 0.01 && Math.abs(confidence - randomConfidence) < 0.015) {
    return {
      type: 'uniform',
      title: 'Suspicious uniform output',
      message:
        'The model returned almost equal probabilities for every class. This usually means the loaded weights do not match the current labels or the model still needs retraining.',
    };
  }

  if (confidence < 0.4) {
    return {
      type: 'low',
      title: 'Low confidence result',
      message:
        'The prediction is uncertain. Try a clearer image with a plain background and more even lighting.',
    };
  }

  return null;
}

function computeStats(history) {
  const total = history.length;
  const counts = {};
  const gradeConfidence = {};

  FALLBACK_CLASSES.forEach((grade) => {
    counts[grade] = 0;
    gradeConfidence[grade] = { sum: 0, peak: 0, count: 0 };
  });

  history.forEach((item) => {
    const grade = String(item.grade || '').toUpperCase();
    if (!counts[grade]) {
      counts[grade] = 0;
      gradeConfidence[grade] = { sum: 0, peak: 0, count: 0 };
    }
    counts[grade] += 1;
    gradeConfidence[grade].sum += item.confidence;
    gradeConfidence[grade].peak = Math.max(gradeConfidence[grade].peak, item.confidence);
    gradeConfidence[grade].count += 1;
  });

  const distribution = Object.entries(counts)
    .filter(([, count]) => count > 0)
    .map(([grade, count]) => ({
      grade,
      count,
      fill: GRADE_COLORS[grade] || '#b98b58',
    }));

  const byGrade = Object.keys(counts).map((grade) => ({
    grade,
    avg: gradeConfidence[grade].count ? gradeConfidence[grade].sum / gradeConfidence[grade].count : 0,
    peak: gradeConfidence[grade].peak,
  }));

  const trend = [...history]
    .reverse()
    .map((item, index) => ({ index: index + 1, confidence: Number(item.confidence.toFixed(3)), grade: item.grade }));

  const avgConfidence = total ? history.reduce((sum, item) => sum + item.confidence, 0) / total : 0;
  const peakConfidence = total ? Math.max(...history.map((item) => item.confidence)) : 0;
  const topGrade = total
    ? Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || null
    : null;
  const premiumCount = (counts.W180 || 0) + (counts.W210 || 0);
  const premiumIndex = total ? Math.round((premiumCount / total) * 100) : 0;
  const passRate = total ? Math.round((history.filter((item) => item.confidence >= 0.7).length / total) * 100) : 0;

  return {
    total,
    distribution,
    byGrade,
    trend,
    avgConfidence,
    peakConfidence,
    topGrade,
    premiumIndex,
    passRate,
  };
}

function App() {
  const labRef = useRef(null);
  const fileInputRef = useRef(null);
  const [health, setHealth] = useState({
    status: 'checking',
    model_loaded: false,
    classes: FALLBACK_CLASSES,
  });
  const [healthError, setHealthError] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    let cancelled = false;

    async function loadHealth() {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) throw new Error(`Health check failed (${response.status})`);
        const data = await response.json();
        if (!cancelled) {
          setHealth({
            status: data.status || 'ok',
            model_loaded: Boolean(data.model_loaded),
            classes: data.classes?.length ? data.classes : FALLBACK_CLASSES,
            ...data,
          });
          setHealthError('');
        }
      } catch (fetchError) {
        if (!cancelled) {
          setHealth({ status: 'offline', model_loaded: false, classes: FALLBACK_CLASSES });
          setHealthError(fetchError.message || 'Backend unavailable');
        }
      }
    }

    loadHealth();
    const intervalId = setInterval(loadHealth, 15000);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl('');
      return undefined;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const activeClasses = health.classes?.length ? health.classes : FALLBACK_CLASSES;

  const probabilityEntries = useMemo(() => {
    const source = prediction?.class_probabilities
      ? Object.entries(prediction.class_probabilities)
      : activeClasses.map((grade) => [grade, 0]);

    return source
      .map(([grade, value]) => ({
        grade: String(grade).toUpperCase(),
        value: Number(value) || 0,
        fill: GRADE_COLORS[String(grade).toUpperCase()] || '#b98b58',
      }))
      .sort((a, b) => b.value - a.value);
  }, [activeClasses, prediction]);

  const diagnosis = useMemo(() => diagnosePrediction(prediction), [prediction]);
  const stats = useMemo(() => computeStats(history), [history]);

  const resultMetrics = useMemo(() => {
    const confidence = prediction?.confidence ?? 0;
    const readiness = health.status === 'ok' && health.model_loaded ? 1 : 0.18;
    const consistency = prediction ? Math.min(Math.max(confidence * 0.92, 0), 1) : 0;
    return [
      { label: 'Classification confidence', value: confidence },
      { label: 'Model readiness', value: readiness },
      { label: 'Consistency estimate', value: consistency },
    ];
  }, [health, prediction]);

  function resetSelection() {
    setSelectedFile(null);
    setPrediction(null);
    setError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  function validateAndSelect(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('Please choose an image smaller than 10MB.');
      return;
    }
    setSelectedFile(file);
    setPrediction(null);
    setError('');
  }

  function handleInputChange(event) {
    validateAndSelect(event.target.files?.[0] || null);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);
    validateAndSelect(event.dataTransfer.files?.[0] || null);
  }

  async function handlePredict() {
    if (!selectedFile) {
      setError('Choose a cashew image first so the grading run can begin.');
      return;
    }
    if (health.status !== 'ok') {
      setError('Backend is offline. Start the FastAPI server and try again.');
      return;
    }
    if (!health.model_loaded) {
      setError('Model weights are not loaded yet. Train the model or add the expected model file first.');
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
        cache: 'no-store',
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Prediction failed.');

      setPrediction(data);
      setHistory((previous) => [
        {
          id: Date.now(),
          fileName: selectedFile.name,
          thumb: previewUrl,
          grade: data.predicted_grade,
          confidence: Number(data.confidence) || 0,
          at: Date.now(),
        },
        ...previous,
      ].slice(0, 12));
    } catch (predictError) {
      setError(predictError.message || 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="status-bar">
        <div className="status-brand">
          <div className="brand-badge">K</div>
          <div>
            <p className="brand-name">
              Kernel<span>IQ</span>
            </p>
            <p className="brand-subtitle">AI cashew quality grading</p>
          </div>
        </div>

        <div className="status-pills">
          <div className={`status-pill ${health.status === 'ok' ? 'ok' : 'bad'}`}>
            {health.status === 'ok' ? <Wifi size={14} /> : <WifiOff size={14} />}
            <span>{health.status === 'ok' ? 'Backend online' : 'Backend offline'}</span>
          </div>
          <div className={`status-pill ${health.model_loaded ? 'ok' : 'warn'}`}>
            <Cpu size={14} />
            <span>{health.model_loaded ? 'Model ready' : 'Model unloaded'}</span>
          </div>
          <a className="status-link" href={`${API_BASE_URL}/docs`} target="_blank" rel="noreferrer">
            API docs <ExternalLink size={13} />
          </a>
        </div>
      </header>

      <main>
        <section className="hero-section">
          <div className="hero-copy">
            <div className="eyebrow">
              <ShieldCheck size={14} />
              <span>AI quality inspection for export-grade cashews</span>
            </div>
            <h1>
              Grade every cashew
              <br />
              with the eye of an <em>expert.</em>
            </h1>
            <p>
              Upload a single kernel image and get an instant W180, W210, W300, or W500 prediction
              with confidence scoring, class probabilities, and live backend diagnostics.
            </p>
            <div className="hero-actions">
              <button type="button" className="primary-button" onClick={() => labRef.current?.scrollIntoView({ behavior: 'smooth' })}>
                Open the grading lab <ArrowRight size={16} />
              </button>
              <a className="secondary-button" href={`${API_BASE_URL}/docs`} target="_blank" rel="noreferrer">
                Explore API docs
              </a>
            </div>
            <div className="hero-stats">
              <div>
                <strong>4</strong>
                <span>export grades</span>
              </div>
              <div>
                <strong>{health.model_loaded ? 'Ready' : 'Pending'}</strong>
                <span>model status</span>
              </div>
              <div>
                <strong>{health.classes?.length || 4}</strong>
                <span>tracked labels</span>
              </div>
            </div>
          </div>

          <div className="hero-visual">
            <div className="hero-card grain">
              <div className="hero-card-glow" />
              <div className="hero-card-top">
                <span className="mini-label">Sample classification</span>
                <span className="mini-badge">
                  <Sparkles size={13} />
                  live lab
                </span>
              </div>
              <div className="hero-grade-row">
                <div>
                  <strong>W210</strong>
                  <p>Jumbo kernel</p>
                </div>
                <span>96.4%</span>
              </div>
              <div className="hero-bars">
                {['W210', 'W180', 'W300', 'W500'].map((grade, index) => (
                  <div key={grade}>
                    <div className="bar-label">
                      <span>{grade}</span>
                      <span>{['96%', '82%', '41%', '14%'][index]}</span>
                    </div>
                    <div className="bar-track">
                      <div
                        className="bar-fill"
                        style={{ width: ['96%', '82%', '41%', '14%'][index], background: GRADE_COLORS[grade] }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="ticker-strip">
          {[
            'Export-grade classification',
            'Session analytics',
            'FastAPI-compatible frontend',
            'Confidence diagnostics',
            'Presentation-ready UI',
          ].map((item) => (
            <span key={item}>
              <ChevronRight size={14} />
              {item}
            </span>
          ))}
        </section>

        <section className="lab-section" ref={labRef}>
          <div className="section-heading">
            <div>
              <p className="section-kicker">Section 02</p>
              <h2>The grading lab</h2>
              <p>Use the updated UI while keeping it aligned with your current FastAPI prediction workflow.</p>
            </div>
            <div className="grade-chips">
              {activeClasses.map((grade) => (
                <span key={grade}>
                  <strong>{grade}</strong>
                  <small>{inferBand(grade)}</small>
                </span>
              ))}
            </div>
          </div>

          <div className="lab-grid">
            <div
              className={`upload-panel ${dragActive ? 'drag-active' : ''}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
            >
              {previewUrl ? (
                <>
                  <img className="preview-image" src={previewUrl} alt={selectedFile?.name || 'Selected cashew'} />
                  {loading && (
                    <div className="scan-overlay">
                      <div className="scan-card">
                        <RefreshCw size={16} className="spin" />
                        <span>Inspecting kernel...</span>
                      </div>
                    </div>
                  )}
                  <button type="button" className="icon-button" onClick={resetSelection} disabled={loading}>
                    <X size={15} />
                  </button>
                  <div className="preview-footer">
                    <span>{selectedFile?.name}</span>
                    <button type="button" onClick={() => fileInputRef.current?.click()}>
                      Replace
                    </button>
                  </div>
                </>
              ) : (
                <button type="button" className="empty-upload" onClick={() => fileInputRef.current?.click()}>
                  <div className="upload-badge">
                    <ImagePlus size={28} />
                  </div>
                  <h3>Drop a kernel image</h3>
                  <p>JPG or PNG up to 10MB. Best results come from clean lighting and a plain background.</p>
                  <span className="primary-button small">
                    <Upload size={15} />
                    Browse files
                  </span>
                </button>
              )}
              <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleInputChange} />
            </div>

            <div className="result-column">
              {error ? (
                <div className="message-panel error">
                  <AlertTriangle size={28} />
                  <h3>Prediction failed</h3>
                  <p>{error}</p>
                </div>
              ) : loading && !prediction ? (
                <div className="message-panel loading">
                  <div className="skeleton hero" />
                  <div className="skeleton line" />
                  <div className="skeleton line short" />
                  <div className="skeleton line" />
                </div>
              ) : prediction ? (
                <>
                  <div className="result-hero grain">
                    <div>
                      <span className="mini-label">Predicted grade</span>
                      <strong data-testid="predicted-grade">{String(prediction.predicted_grade).toUpperCase()}</strong>
                      <p data-testid="business-band">{inferBand(prediction.predicted_grade)}</p>
                      <small>{GRADE_INFO[String(prediction.predicted_grade).toUpperCase()]?.description || 'Predicted class from the current model.'}</small>
                    </div>
                    <div className="confidence-box">
                      <CheckCircle2 size={22} />
                      <span>Confidence</span>
                      <strong>{formatPercent(prediction.confidence)}</strong>
                    </div>
                  </div>

                  <div className="result-card">
                    <div className="card-heading">
                      <div>
                        <h3>Class probabilities</h3>
                        <p>Ranked from most likely to least likely</p>
                      </div>
                      <BarChart3 size={16} />
                    </div>
                    <div className="probability-list">
                      {probabilityEntries.map((entry, index) => (
                        <div key={entry.grade}>
                          <div className="probability-row">
                            <div>
                              <strong>{entry.grade}</strong>
                              <span>{inferBand(entry.grade)}</span>
                            </div>
                            <b>{formatPercent(entry.value)}</b>
                          </div>
                          <div className="bar-track">
                            <div
                              className={`bar-fill ${index === 0 ? 'primary' : ''}`}
                              style={{ width: `${Math.max(entry.value * 100, 3)}%`, background: entry.fill }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {diagnosis && (
                    <div className={`diagnostic-card ${diagnosis.type}`}>
                      <AlertTriangle size={18} />
                      <div>
                        <strong>{diagnosis.title}</strong>
                        <p>{diagnosis.message}</p>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="message-panel empty">
                  <Sparkles size={28} />
                  <h3>Awaiting your sample</h3>
                  <p>Upload a kernel image to reveal its predicted grade, confidence, and class breakdown.</p>
                </div>
              )}

              <div className="status-card">
                <div className="card-heading">
                  <div>
                    <h3>System status</h3>
                    <p>Live backend and model diagnostics</p>
                  </div>
                  <Activity size={16} />
                </div>
                <div className="status-grid">
                  <div>
                    <span>Backend</span>
                    <strong>{health.status === 'ok' ? 'Connected' : 'Offline'}</strong>
                  </div>
                  <div>
                    <span>Model</span>
                    <strong>{health.model_loaded ? 'Loaded' : 'Missing'}</strong>
                  </div>
                  <div>
                    <span>Classes</span>
                    <strong>{activeClasses.length}</strong>
                  </div>
                  <div>
                    <span>API host</span>
                    <strong>{API_BASE_URL.replace(/^https?:\/\//, '')}</strong>
                  </div>
                </div>
                {healthError && <p className="status-note">{healthError}</p>}
                {!selectedFile ? (
                  <button type="button" className="primary-button full" onClick={() => fileInputRef.current?.click()}>
                    Load image
                  </button>
                ) : (
                  <button type="button" className="primary-button full" onClick={handlePredict} disabled={loading}>
                    {loading ? 'Running grading...' : 'Start grading run'}
                  </button>
                )}
              </div>
            </div>
          </div>
        </section>

        <section className="analytics-section">
          <div className="section-heading">
            <div>
              <p className="section-kicker">Section 03</p>
              <h2>Inspection analytics</h2>
              <p>Session-only telemetry for the kernels processed in this frontend run.</p>
            </div>
          </div>

          <div className="kpi-grid">
            <div className="kpi-card">
              <LayersIcon />
              <strong>{stats.total}</strong>
              <span>samples graded</span>
            </div>
            <div className="kpi-card accent">
              <Gauge size={18} />
              <strong>{formatPercent(stats.avgConfidence)}</strong>
              <span>average confidence</span>
            </div>
            <div className="kpi-card">
              <ShieldCheck size={18} />
              <strong>{stats.topGrade || '—'}</strong>
              <span>top grade this session</span>
            </div>
            <div className="kpi-card">
              <Activity size={18} />
              <strong>{stats.passRate}%</strong>
              <span>passes above 70%</span>
            </div>
          </div>

          <div className="chart-grid">
            <div className="chart-card">
              <div className="card-heading">
                <div>
                  <h3>Grade distribution</h3>
                  <p>Share of processed samples</p>
                </div>
              </div>
              <div className="chart-area">
                {stats.distribution.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={stats.distribution} dataKey="count" nameKey="grade" innerRadius={54} outerRadius={84} paddingAngle={4}>
                        {stats.distribution.map((entry) => (
                          <Cell key={entry.grade} fill={entry.fill} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => value} />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <EmptyChart message="Grade a few kernels to populate this chart." />
                )}
              </div>
            </div>

            <div className="chart-card wide">
              <div className="card-heading">
                <div>
                  <h3>Confidence by grade</h3>
                  <p>Average and peak confidence for each class</p>
                </div>
              </div>
              <div className="chart-area">
                {stats.total ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={stats.byGrade} margin={{ top: 12, right: 8, left: -12, bottom: 0 }}>
                      <CartesianGrid stroke="rgba(109, 86, 58, 0.12)" vertical={false} />
                      <XAxis dataKey="grade" tickLine={false} axisLine={false} />
                      <YAxis tickFormatter={(value) => `${Math.round(value * 100)}%`} tickLine={false} axisLine={false} domain={[0, 1]} />
                      <Tooltip formatter={(value) => formatPercent(value)} />
                      <Bar dataKey="avg" radius={[8, 8, 0, 0]}>
                        {stats.byGrade.map((entry) => (
                          <Cell key={entry.grade} fill={GRADE_COLORS[entry.grade] || '#b98b58'} />
                        ))}
                      </Bar>
                      <Bar dataKey="peak" radius={[8, 8, 0, 0]} fillOpacity={0.35}>
                        {stats.byGrade.map((entry) => (
                          <Cell key={`${entry.grade}-peak`} fill={GRADE_COLORS[entry.grade] || '#b98b58'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <EmptyChart message="Confidence trends appear once predictions are logged." />
                )}
              </div>
            </div>

            <div className="chart-card wide">
              <div className="card-heading">
                <div>
                  <h3>Confidence trend</h3>
                  <p>Latest predictions in session order</p>
                </div>
              </div>
              <div className="chart-area">
                {stats.trend.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={stats.trend} margin={{ top: 12, right: 8, left: -12, bottom: 0 }}>
                      <defs>
                        <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#d28a22" stopOpacity={0.55} />
                          <stop offset="100%" stopColor="#d28a22" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="rgba(109, 86, 58, 0.12)" vertical={false} />
                      <XAxis dataKey="index" tickLine={false} axisLine={false} />
                      <YAxis tickFormatter={(value) => `${Math.round(value * 100)}%`} tickLine={false} axisLine={false} domain={[0, 1]} />
                      <Tooltip formatter={(value) => formatPercent(value)} />
                      <Area type="monotone" dataKey="confidence" stroke="#d28a22" strokeWidth={3} fill="url(#trendFill)" />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <EmptyChart message="Run a few predictions to see the trend line." />
                )}
              </div>
            </div>

            <div className="chart-card">
              <div className="card-heading">
                <div>
                  <h3>Premium index</h3>
                  <p>Share of W180 and W210 samples</p>
                </div>
              </div>
              <div className="radial-wrap">
                {stats.total ? (
                  <>
                    <div className="radial-chart">
                      <ResponsiveContainer width="100%" height="100%">
                        <RadialBarChart innerRadius="68%" outerRadius="100%" data={[{ value: stats.premiumIndex }]} startAngle={90} endAngle={-270}>
                          <RadialBar dataKey="value" cornerRadius={14} fill="#d28a22" background={{ fill: 'rgba(143, 107, 70, 0.14)' }} />
                        </RadialBarChart>
                      </ResponsiveContainer>
                      <div className="radial-center">
                        <strong>{stats.premiumIndex}%</strong>
                        <span>premium</span>
                      </div>
                    </div>
                    <div className="metric-list">
                      {resultMetrics.map((metric) => (
                        <div key={metric.label}>
                          <div className="metric-row">
                            <span>{metric.label}</span>
                            <strong>{formatPercent(metric.value)}</strong>
                          </div>
                          <div className="bar-track">
                            <div className="bar-fill" style={{ width: `${metric.value * 100}%`, background: '#d28a22' }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <EmptyChart message="Premium index is calculated from session history." />
                )}
              </div>
            </div>
          </div>
        </section>

        <section className="history-section">
          <div className="section-heading">
            <div>
              <p className="section-kicker">Section 04</p>
              <h2>Recent grading activity</h2>
              <p>Quick access to the latest sample results from this session.</p>
            </div>
          </div>

          {history.length ? (
            <div className="history-grid">
              {history.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className="history-card"
                  onClick={() => {
                    setPreviewUrl(item.thumb);
                    setPrediction({
                      predicted_grade: item.grade,
                      confidence: item.confidence,
                      class_probabilities: { [item.grade]: item.confidence },
                    });
                    setError('');
                    labRef.current?.scrollIntoView({ behavior: 'smooth' });
                  }}
                >
                  {item.thumb ? <img src={item.thumb} alt={item.fileName} /> : <div className="history-fallback" />}
                  <div className="history-overlay">
                    <strong>{item.grade}</strong>
                    <span>{formatPercent(item.confidence)}</span>
                    <small>
                      <Clock3 size={11} />
                      {timeAgo(item.at)}
                    </small>
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <div className="history-empty">
              <Clock3 size={24} />
              <p>No samples graded yet</p>
              <span>Your recent inspection log will appear here as predictions are completed.</span>
            </div>
          )}
        </section>

        <footer className="app-footer">
          <p>KernelIQ frontend adapted for your FastAPI backend.</p>
          <span>{formatTime()}</span>
        </footer>
      </main>
    </div>
  );
}

function LayersIcon() {
  return <Gauge size={18} />;
}

function EmptyChart({ message }) {
  return (
    <div className="empty-chart">
      <BarChart3 size={22} />
      <p>{message}</p>
    </div>
  );
}

export default App;
