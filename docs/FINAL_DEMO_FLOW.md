# SanDisk Cerebrum: Final Demo Flow

## Complete End-to-End Demonstration Journey

The Cerebrum platform is organized around a single unified engineering workflow:
$$\text{RISK} \longrightarrow \text{LOCATION} \longrightarrow \text{EVIDENCE} \longrightarrow \text{DECISION} \longrightarrow \text{YIELD IMPACT} \longrightarrow \text{ACTION}$$

---

### Step 1: Fab Operations Command Center (Executive View)
- **Route**: `/command-center`
- **What to Observe**:
  - Top Hero Metric Tiles: 85.22% Precision, 37.17% Recall, 0.5177 F1, 93.78% Compute Saved.
  - Population Switcher: Switch between Independent 40-Wafer Validation (EXP-04) and Canonical 32-Wafer Confirmation (EXP-03B).
  - Yield-at-Risk Panel: View predicted high-risk dies (602 dies / 1.85% exposure) vs observed yield (95.77%).
  - Model Health & Trust Panel: Confirm Model B2 frozen status and 38/38 passing unit tests.
  - Judge CTA Banner: Prominent "OPEN LIVE INVESTIGATION" button. Click to launch into Wafer `W_N_0057` Die `R32/C23`.

---

### Step 2: Spatial Wafer Intelligence & Hotspots
- **Route**: `/wafer-intelligence/W_N_0057`
- **What to Observe**:
  - Wafer Header: 835 total dies, 34 actual defects (4.07% defect rate), 32 cascade detections, +2 rescued defects.
  - Spatial Hotspot Clusters Panel: Observe the wafer automatically partitioned into:
    - Hotspot A: Defect Epicenter (Centroid R32, C22, 18 high-risk dies, 78.4% mean risk).
    - Hotspot B: Edge Bevel Ring (Centroid R12, C48, 8 high-risk dies).
    - Hotspot C: Center Core (Centroid R28, C26, 6 high-risk dies).
  - Interactive Wafer Map: Toggle between Status, Heatmap, Uncertainty, and Rescued Defect views.

---

### Step 3: Die Forensic Workstation & Multi-Sensor Evidence
- **Route**: `/wafer-intelligence/W_N_0057/32/23`
- **What to Observe**:
  - Die Status: R32 / C23 • VERDICT: FAIL • Raw P: 76.51% • Calibrated Risk: 72.84% • RESCUED DEFECT.
  - Tab 1: **Evidence Chain** ("Why Cerebrum Classified This Die As High Risk"):
    - Electrical Evidence: Peak signal 112.4 mV (> 104.0 threshold), 8 anomalous blocks.
    - Spatial Evidence: Radial distance 0.4850, Core pattern, defect cluster proximity.
    - Signal Evidence: Secondary inspection triggered in Gate B [0.295, 0.550].
    - Decision Trace: Model A initially predicted 0.4210 (Escape). Model B2 resolved sub-die anomalies at 0.7651 (True Defect Rescued).
  - Tab 2: **Decision Trace**: Detailed 3-stage cascade flow.
  - Tab 3: **TCN Forensics**:
    - Multi-Scale Dilated TCN Sequence Attribution.
    - Anomaly Span: Blocks 840 – 1,020 highlighted with saliency gradients.
    - Explicit Banner: *FORENSIC ANALYSIS — NOT USED FOR PRODUCTION ROUTING*.
  - Tab 4: **2,000-Block Signal**: Interactive zoomable raw waveform trace.
  - Tab 5: **Local 3×3 Neighborhood**: Shows defect clustering around row 32, col 23.

---

### Step 4: What-If Inspection Policy Simulator
- **Route**: `/yield-analytics`
- **What to Observe**:
  - Safety Label: *WHAT-IF SIMULATION — DOES NOT CHANGE PRODUCTION POLICY*.
  - Dynamic Sliders:
    - Model A Threshold ($\tau_A$): Adjust between 0.30 and 0.70.
    - Uncertainty Gate Window: Adjust lower and upper bounds.
    - Model B2 Threshold ($\tau_B$): Adjust between 0.30 and 0.70.
    - Escalation Budget Cap: Set a tester throughput ceiling (e.g., 8.0%).
  - Live Vectorized Feedback (<30ms on 27,972 dies):
    - Escalation Rate vs Compute Saved trade-off.
    - Precision, Recall, F1, MCC, and Net TP differences vs frozen B2 baseline.
  - Reset Button: Instantly snaps back to frozen production configuration ($\tau_A=0.5100, \text{Gate}=[0.295, 0.550], \tau_B=0.5376$).

---

### Step 5: Model Evolution & Scientific Transparency
- **Route**: `/model-lab`
- **What to Observe**:
  - Evidence-Driven Evolution Panel: Clear history explaining why plausible avenues were rejected:
    - EXP-05 (Late Stacking): REJECTED (distorted gating).
    - EXP-06 (XGBoost/CatBoost Ensemble): REJECTED (3.8× latency, negligible F1 gain).
    - EXP-07 (FFT/Kurtosis): REJECTED (feature redundancy, -0.0038 F1).
    - EXP-08 (Multi-Scale TCN): RETAINED FOR FORENSICS (waveform saliency).
    - EXP-09 (Isotonic Calibration): RETAINED FOR RISK REPORTING (ECE 0.0035, -97.2%).
    - EXP-10 (Context-Awareness): REJECTED (overfitting variance).
  - Technical Architecture Diagram: Visual representation of production decision vs waveform forensics vs risk reporting.
  - Live Dynamic Evaluation Engine: Run real-time verification across 27,972 canonical validation dies with exact 0.000000 metric reproducibility.
