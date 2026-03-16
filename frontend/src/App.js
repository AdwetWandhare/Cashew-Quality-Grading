import React from 'react';
import { 
  Database, Play, LayoutGrid, CheckCircle, 
  Settings, Search, ArrowRight, Info, AlertTriangle 
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  PieChart, Pie, Cell, Legend 
} from 'recharts';

// --- Mock Data ---
const barData = [
  { name: 'W180', value: 55000, fill: '#3498db' },
  { name: 'W210', value: 75000, fill: '#2980b9' },
  { name: 'W240', value: 60000, fill: '#f39c12' },
  { name: 'W320', value: 105000, fill: '#e67e22' },
  { name: 'Others', value: 30000, fill: '#e74c3c' },
];

const pieData = [
  { name: 'W180', value: 15, color: '#3498db' },
  { name: 'W210', value: 20, color: '#2980b9' },
  { name: 'W240', value: 25, color: '#f39c12' },
  { name: 'W320', value: 30, color: '#e67e22' },
  { name: 'Defective', value: 10, color: '#e74c3c' },
];

const App = () => {
  return (
    <div className="min-h-screen bg-[#0B1120] text-gray-200 p-6 font-sans">
      {/* HEADER */}
      <header className="flex justify-between items-center mb-8 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <span className="text-orange-500 text-3xl">🥜</span> 
            CASHEW QUALITY ANALYTICS & GRADING SYSTEM
          </h1>
        </div>
        <div className="text-right">
          <p className="text-sm font-semibold text-slate-400">Dataset and Batch Processing Analytics</p>
          <p className="text-xs text-slate-500 uppercase tracking-widest">Dataset-Driven Grading and Deep Learning Analysis</p>
        </div>
      </header>

      {/* MAIN GRID */}
      <div className="grid grid-cols-12 gap-6">
        
        {/* LEFT: DATASET MANAGEMENT */}
        <div className="col-span-3 space-y-6">
          <section className="bg-slate-900/50 p-5 rounded-xl border border-slate-800">
            <h2 className="text-xs font-bold uppercase text-slate-400 mb-4 tracking-wider">Dataset Management</h2>
            <button className="w-full bg-orange-600 hover:bg-orange-700 text-white font-bold py-3 rounded-lg flex items-center justify-center gap-2 mb-3 transition-all">
              <Database size={18} /> LOAD DATASET
            </button>
            <button className="w-full border border-slate-700 hover:bg-slate-800 py-3 rounded-lg flex items-center justify-center gap-2 mb-3">
              <Search size={18} /> SELECT BATCH
            </button>
            <button className="w-full border-2 border-orange-600/50 hover:bg-orange-600/10 text-orange-500 py-3 rounded-lg flex items-center justify-center gap-2">
              <Play size={18} /> START BATCH PROCESSING
            </button>

            <div className="mt-8 space-y-4">
              <div className="flex items-center gap-3">
                <LayoutGrid className="text-slate-500" />
                <div>
                  <p className="text-xs text-slate-500 uppercase">Dataset Size (kernels)</p>
                  <p className="text-xl font-bold">125,780</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Settings className="text-slate-500" />
                <div>
                  <p className="text-xs text-slate-500 uppercase">Current Batch</p>
                  <p className="text-lg font-bold text-green-500 italic">batch_004</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <CheckCircle className="text-green-500" />
                <div>
                  <p className="text-xs text-slate-500 uppercase">Processing Status</p>
                  <p className="text-xl font-bold">Completed</p>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* MIDDLE: GALLERY & ANNOTATION */}
        <div className="col-span-6 bg-slate-900/50 p-5 rounded-xl border border-slate-800">
          <h2 className="text-xs font-bold uppercase text-slate-400 mb-4 tracking-wider">Dataset Gallery & Annotation</h2>
          <div className="grid grid-cols-6 gap-2">
            {[...Array(24)].map((_, i) => (
              <div key={i} className={`relative rounded border ${i === 10 ? 'border-orange-500 ring-2 ring-orange-500/50 scale-110 z-10' : 'border-slate-700'} aspect-square bg-slate-800 overflow-hidden`}>
                <div className="absolute top-1 right-1 px-1 bg-blue-600 text-[8px] rounded">W180</div>
                {/* Image Placeholder */}
                <div className="w-full h-full flex items-center justify-center opacity-40">🥜</div>
                {i === 10 && (
                   <div className="absolute bottom-0 left-0 right-0 bg-slate-900/90 text-[8px] p-1 text-center">
                    GT: W320, PRED: W320 (98.5%)
                   </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT: GRADING STATISTICS */}
        <div className="col-span-3 space-y-6">
          <section className="bg-slate-900/50 p-5 rounded-xl border border-slate-800 h-full">
            <h2 className="text-xs font-bold uppercase text-slate-400 mb-4 tracking-wider">Grading Statistics</h2>
            
            <div className="text-center mb-4">
              <p className="text-xs text-slate-500 uppercase">Total Dataset Kernels</p>
              <h3 className="text-4xl font-bold tracking-tighter">125,780</h3>
            </div>

            <div className="h-48 mb-8">
               <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData}>
                  <Bar dataKey="value" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={pieData} innerRadius={40} outerRadius={60} dataKey="value">
                    {pieData.map((entry, index) => <Cell key={index} fill={entry.color} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </section>
        </div>

        {/* BOTTOM ROW: PIPELINE & LOGS */}
        <div className="col-span-4 bg-slate-900/50 p-5 rounded-xl border border-slate-800">
          <h2 className="text-xs font-bold uppercase text-slate-400 mb-6 tracking-wider">Processing Pipeline</h2>
          <div className="flex items-center justify-between text-[10px] text-center uppercase font-bold text-slate-500">
            <div className="flex flex-col items-center gap-2">
              <div className="p-3 border border-blue-500 rounded bg-blue-500/10 text-blue-400"><Database size={20}/></div>
              <span>Import</span>
            </div>
            <ArrowRight size={14}/>
            <div className="flex flex-col items-center gap-2">
              <div className="p-3 border border-blue-500 rounded bg-blue-500/10 text-blue-400"><Settings size={20}/></div>
              <span>Preprocessing</span>
            </div>
            <ArrowRight size={14}/>
            <div className="flex flex-col items-center gap-2">
              <div className="p-3 border border-orange-500 rounded bg-orange-500/10 text-orange-400"><Info size={20}/></div>
              <span>Balancing</span>
            </div>
            <ArrowRight size={14}/>
            <div className="flex flex-col items-center gap-2">
              <div className="p-3 border border-orange-500 rounded bg-orange-500/10 text-orange-400"><AlertTriangle size={20}/></div>
              <span>Evaluation</span>
            </div>
          </div>
        </div>

        <div className="col-span-5 bg-slate-900/50 p-5 rounded-xl border border-slate-800">
          <h2 className="text-xs font-bold uppercase text-slate-400 mb-4 tracking-wider">Batch Processing Log</h2>
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-slate-800">
                <th className="pb-2">FILE NAME</th>
                <th className="pb-2 text-center">PREDICTED</th>
                <th className="pb-2 text-center">CONF (%)</th>
                <th className="pb-2 text-center">TIME (s)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {[1, 2, 3, 4, 5].map((i) => (
                <tr key={i} className="hover:bg-white/5">
                  <td className="py-2 text-slate-300">k_img_00{i}.jpg</td>
                  <td className="py-2 text-center font-bold">W180</td>
                  <td className="py-2 text-center">98.5</td>
                  <td className="py-2 text-center">0.12</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="col-span-3 bg-slate-900/50 p-5 rounded-xl border border-slate-800">
          <h2 className="text-xs font-bold uppercase text-slate-400 mb-4 tracking-wider">Overall Dataset Confidence</h2>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-[10px] mb-1 uppercase font-bold">
                <span>Classification Confidence</span>
                <span className="text-orange-500">92%</span>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                <div className="bg-orange-500 h-full w-[92%]"></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-[10px] mb-1 uppercase font-bold">
                <span>Ground-Truth Consistency</span>
                <span className="text-blue-500">88%</span>
              </div>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                <div className="bg-blue-500 h-full w-[88%]"></div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default App;