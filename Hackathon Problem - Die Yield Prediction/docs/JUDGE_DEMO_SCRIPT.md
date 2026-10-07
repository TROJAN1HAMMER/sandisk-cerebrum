# SanDisk Cerebrum: Judge Presentation Script

## Speaker Script & Demonstration Timing (~5 to 7 Minutes)

### Phase 1: Product Positioning & Executive Overview (1 Minute)
> "Good morning, judges. Today we are presenting **Cerebrum**, SanDisk's selective semiconductor yield-intelligence platform.
>
> In semiconductor manufacturing, predicting die failure is only half the battle. A fab engineer cannot afford to inspect all 2,000 blocks across 30,000 dies per wafer—it would bottleneck the entire test floor.
>
> Cerebrum solves this by acting as a selective multi-resolution decision engine:
> 1. It screens 100% of dies with an ultra-fast baseline model.
> 2. It bypasses 93.78% of dies on the fast path without secondary testing.
> 3. It escalates only the 6.22% borderline, uncertain dies to an enhanced sub-die model.
>
> Here in the Command Center, you can see our independently validated results across 40 unseen wafers and 32,598 dies:
> - **85.22% Precision** with only 89 false alarms.
> - **93.78% Compute Saved**.
> - **23.30% False Negative Rescue Rate**, adding +37 net defect recoveries that standard screening missed entirely.
>
> We don't claim hypothetical factory dollar savings because the competition dataset doesn't contain fab wafer-start cost metrics. Instead, we show a concrete, rigorous **Yield-at-Risk** indicator quantifying spatial defect exposure."

---

### Phase 2: Live Case Investigation (2 Minutes)
> "Let's demonstrate how an engineer uses Cerebrum in production. I'll click **'OPEN LIVE INVESTIGATION'** to examine wafer `W_N_0057`.
>
> Notice what happens immediately:
> - Cerebrum partitions the wafer into physical **Spatial Hotspots**: an Edge Bevel Ring, a Center Core, and a concentrated Defect Epicenter.
> - Let's drill into die `R32 / C23`.
>
> In our **Die Forensic Workstation**, Cerebrum does not simply output a black-box label. It presents a structured **Multi-Sensor Evidence Chain**:
> 1. **Electrical Evidence**: Peak voltage reading reached 112.4 mV, with 8 sequential blocks exceeding the 104.0 threshold.
> 2. **Spatial Evidence**: Normalized radial distance is 0.4850, located within a known spatial defect cluster.
> 3. **Decision Trace**: Model A initially scored this die at 0.4210—meaning standalone screening would have let this defect escape! But because 0.4210 fell into our uncertainty gate [0.295, 0.550], Cerebrum escalated it to Model B2. Model B2 inspected the sub-die block readings and correctly classified it as a defect (P=0.7651), recovering the true failure.
> 4. **TCN Waveform Forensics**: Here under the TCN Forensics tab, our multi-scale dilated temporal convolutional network localizes the exact anomaly window to blocks 840 through 1,020. Notice our explicit engineering disclaimer: the TCN is utilized purely for forensic waveform attribution, without contaminating the frozen production cascade decision."

---

### Phase 3: What-If Policy Simulation (1.5 Minutes)
> "Now let's look at fab operational flexibility in **Yield Analytics**.
>
> In real manufacturing, test budgets change. A yield engineer might have 10% tester capacity on Monday, but only 5% on Friday.
>
> We built the **What-If Policy Simulator**. Operating in-memory across all 27,972 canonical validation dies, an engineer can adjust:
> - Model A threshold ($\tau_A$)
> - Uncertainty gate lower and upper bounds
> - Model B2 threshold ($\tau_B$)
> - Or set a strict Secondary Analysis Budget Cap.
>
> In under 30 milliseconds, the simulator recalculates the entire confusion matrix, precision, recall, and compute savings. Notice our safety banner: *WHAT-IF SIMULATION DOES NOT CHANGE PRODUCTION POLICY*. It allows test engineers to explore throughput vs recall trade-offs safely before committing policy changes."

---

### Phase 4: Model Evolution & Scientific Integrity (1 Minute)
> "Finally, let's visit the **Model Lab**.
>
> When improving an ML system, it is tempting to stack ensembles or add complex features blindly. In Cerebrum, we followed an evidence-driven scientific approach across experiments EXP-05 through EXP-10:
> - We tested **Late Probability Stacking (EXP-05)**: REJECTED because it distorted probability calibration and broke cascade gating.
> - We tested an **XGBoost + CatBoost Ensemble (EXP-06)**: REJECTED because a 3.8× inference latency increase yielded only +0.0011 F1 gain.
> - We tested **Spectral FFT Harmonics (EXP-07)**: REJECTED because it added redundancy and degraded F1.
> - We tested **Context-Aware Spatial Aggregations (EXP-10)**: REJECTED because it introduced wafer-level variance overfitting.
> - What we retained was **Isotonic Calibration (EXP-09)** for risk reporting (reducing ECE by 97.2%) and **Multi-Scale TCN (EXP-08)** for forensic waveform saliency.
>
> Every metric shown is backed by independent validation, zero data leakage, and 38 passing unit tests.
>
> Cerebrum is not just an ML experiment—it is a production-grade semiconductor yield engineering platform. Thank you, and we welcome your questions."
