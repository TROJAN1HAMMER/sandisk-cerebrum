import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppShell } from './layouts/AppShell';
import { CommandCenter } from './pages/CommandCenter';
import { YieldAnalytics } from './pages/YieldAnalytics';
import { WaferIntelligence } from './pages/WaferIntelligence';
import { CascadeDiagnostics } from './pages/CascadeDiagnostics';
import { ModelLab } from './pages/ModelLab';
import { ForensicAudit } from './pages/ForensicAudit';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppShell />}>
          <Route index element={<Navigate to="/command-center" replace />} />
          <Route path="command-center" element={<CommandCenter />} />
          <Route path="yield-analytics" element={<YieldAnalytics />} />
          <Route path="wafer-intelligence" element={<WaferIntelligence />} />
          <Route path="wafer-intelligence/:waferId" element={<WaferIntelligence />} />
          <Route path="wafer-intelligence/:waferId/:row/:col" element={<WaferIntelligence />} />
          <Route path="cascade-diagnostics" element={<CascadeDiagnostics />} />
          <Route path="model-lab" element={<ModelLab />} />
          <Route path="forensic-audit" element={<ForensicAudit />} />
          <Route path="*" element={<Navigate to="/command-center" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};


export default App;
