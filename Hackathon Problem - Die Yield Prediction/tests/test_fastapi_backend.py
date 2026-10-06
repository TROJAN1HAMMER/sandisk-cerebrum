import os
import io
import json
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.deps import init_app_state, state
from src.live_inference import predict_single_die_live

class TestFastAPIBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize app state once
        init_app_state()
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        """GET /api/v1/system/health returns online status."""
        resp = self.client.get("/api/v1/system/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["app_name"], "SanDisk Cerebrum API")

    def test_02_status_endpoint(self):
        """GET /api/v1/system/status returns verified engine parameters."""
        resp = self.client.get("/api/v1/system/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["models_loaded"])
        self.assertTrue(data["model_a_active"])
        self.assertTrue(data["model_b_active"])
        self.assertEqual(data["total_validation_dies"], 27972)
        self.assertEqual(data["validation_wafers_count"], 32)
        self.assertIn("128 Train / 32 Val", data["canonical_split"])
        self.assertEqual(data["gate_b_window"], "[0.25, 0.55]")

    def test_03_authoritative_metrics(self):
        """GET /api/v1/models/metrics returns exact verified single source of truth."""
        resp = self.client.get("/api/v1/models/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        
        # Verify Model A authoritative metrics
        ma = data["model_A"]
        self.assertEqual(ma["f1"], 0.4907)
        self.assertEqual(ma["auc_pr"], 0.4498)
        self.assertEqual(ma["precision"], 0.8947)
        self.assertEqual(ma["recall"], 0.3381)
        self.assertEqual(ma["confusion_matrix"]["tp"], 357)
        self.assertEqual(ma["confusion_matrix"]["fn"], 699)

        # Verify Cascade Gate B authoritative metrics
        gb = data["cascade_gate_B"]
        self.assertEqual(gb["f1"], 0.5072)
        self.assertEqual(gb["auc_pr"], 0.4677)
        self.assertEqual(gb["precision"], 0.8138)
        self.assertEqual(gb["recall"], 0.3684)
        self.assertEqual(gb["escalation_count"], 3317)
        self.assertEqual(gb["compute_saved_pct"], 88.14)
        self.assertEqual(gb["confusion_matrix"]["tp"], 389)
        self.assertEqual(gb["confusion_matrix"]["fn"], 667)
        self.assertEqual(gb["fn_recovery_metrics"]["gross_fn_recovered"], 37)
        self.assertEqual(gb["fn_recovery_metrics"]["net_tp_gain"], 32)

    def test_04_benchmark_and_sweeps(self):
        """GET /api/v1/models/benchmark returns comparison table and Pareto sweep."""
        resp = self.client.get("/api/v1/models/benchmark")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(len(data["benchmark_rows"]), 0)
        self.assertGreater(len(data["gate_sweep_rows"]), 0)

    def test_05_error_drilldown(self):
        """GET /api/v1/models/error-drilldown returns exactly 37 recovered FNs."""
        resp = self.client.get("/api/v1/models/error-drilldown?group=recovered_fns&limit=50")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_in_group"], 37)
        self.assertEqual(data["returned_count"], 37)
        self.assertIn("wafer_id", data["records"][0])
        self.assertIn("die_row", data["records"][0])

    def test_05b_analytical_curves_and_distributions(self):
        """GET /api/v1/models/probability-distribution, /roc-pr-curves, /high-risk-dies."""
        resp_hist = self.client.get("/api/v1/models/probability-distribution")
        self.assertEqual(resp_hist.status_code, 200)
        h_data = resp_hist.json()
        self.assertEqual(len(h_data["bin_labels"]), 20)
        self.assertEqual(sum(h_data["model_a_counts"]), 27972)
        self.assertEqual(sum(h_data["cascade_counts"]), 27972)

        resp_curves = self.client.get("/api/v1/models/roc-pr-curves")
        self.assertEqual(resp_curves.status_code, 200)
        c_data = resp_curves.json()
        self.assertAlmostEqual(c_data["roc"]["model_A"]["auc"], 0.7934, places=3)
        self.assertAlmostEqual(c_data["roc"]["cascade"]["auc"], 0.7990, places=3)
        self.assertAlmostEqual(c_data["pr"]["model_A"]["auc_pr"], 0.4498, places=3)
        self.assertAlmostEqual(c_data["pr"]["cascade"]["auc_pr"], 0.4677, places=3)

        resp_risk = self.client.get("/api/v1/models/high-risk-dies?limit=10")
        self.assertEqual(resp_risk.status_code, 200)
        r_data = resp_risk.json()
        self.assertEqual(len(r_data["records"]), 10)
        self.assertGreaterEqual(r_data["records"][0]["prob_Cascade_B"], r_data["records"][9]["prob_Cascade_B"])

    def test_06_wafers_list_and_detail(self):
        """GET /api/v1/wafers returns all 32 holdout wafers."""
        resp = self.client.get("/api/v1/wafers")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_wafers"], 32)
        self.assertEqual(data["total_holdout_dies"], 27972)

        # Query single wafer detail and dies
        wid = "W_N_0057"
        resp_dies = self.client.get(f"/api/v1/wafers/{wid}/dies?filter_mode=all&limit=500")
        self.assertEqual(resp_dies.status_code, 200)
        w_data = resp_dies.json()
        self.assertEqual(w_data["summary"]["wafer_id"], wid)
        self.assertGreater(len(w_data["dies"]), 0)

        # Query single die exact endpoint
        resp_single = self.client.get(f"/api/v1/wafers/{wid}/dies/32/23")
        self.assertEqual(resp_single.status_code, 200)
        s_die = resp_single.json()
        self.assertEqual(s_die["wafer_id"], wid)
        self.assertEqual(s_die["die_row"], 32)
        self.assertEqual(s_die["die_col"], 23)
        self.assertEqual(s_die["actual_label"], 1)
        self.assertEqual(s_die["recovered_FN_Gate_B"], 1)

    def test_07_demo_samples_route(self):
        """GET /api/v1/inference/demo-samples returns verified demo items."""
        resp = self.client.get("/api/v1/inference/demo-samples")
        self.assertEqual(resp.status_code, 200)
        demos = resp.json()
        self.assertGreater(len(demos), 0)
        self.assertTrue(any(d["sample_id"] == "sample_1_recovered_false_negative" for d in demos))

    def test_08_single_die_inference_fast_path(self):
        """POST /api/v1/inference/die correctly routes confident pass to FAST PATH."""
        # W_N_0001 Row 0 Col 13 (Demo 3: Confident Pass, index 0 of validation set)
        req_body = {
            "wafer_id": "W_N_0001",
            "die_row": 0,
            "die_col": 13
        }
        resp = self.client.post("/api/v1/inference/die", json=req_body)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertEqual(res["compute_path"], "FAST PATH")
        self.assertFalse(res["in_gate"])
        self.assertEqual(res["gate_status"], "BYPASSED")
        self.assertEqual(res["final_decision"], 0)
        self.assertIsNone(res["prob_Model_B"])
        self.assertIsNotNone(res["latency_ms"])

    def test_09_single_die_inference_escalated_path(self):
        """POST /api/v1/inference/die correctly escalates ambiguous die to Model B and recovers defect."""
        # W_N_0057 Row 32 Col 23 (Demo 1: Recovered Failure, index 3736)
        req_body = {
            "wafer_id": "W_N_0057",
            "die_row": 32,
            "die_col": 23
        }
        resp = self.client.post("/api/v1/inference/die", json=req_body)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertEqual(res["compute_path"], "ESCALATED PATH")
        self.assertTrue(res["in_gate"])
        self.assertEqual(res["gate_status"], "ESCALATED")
        self.assertTrue(res["model_B_invoked"])
        self.assertIsNotNone(res["prob_Model_B"])
        self.assertGreaterEqual(res["prob_Model_B"], 0.5376)
        self.assertEqual(res["final_decision"], 1)

    def test_10_batch_inference_valid_and_invalid(self):
        """POST /api/v1/inference/batch runs batch prediction and validates CSV schema."""
        # 1. Valid CSV
        cols = ["wafer_id", "die_row", "die_col"] + [f"feature_{i}" for i in range(1, 501)]
        df_dummy = pd.DataFrame(np.zeros((5, len(cols))), columns=cols)
        df_dummy["wafer_id"] = "W_TEST_0001"
        csv_buffer = io.BytesIO()
        df_dummy.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        resp = self.client.post(
            "/api/v1/inference/batch",
            files={"file": ("test_dies.csv", csv_buffer, "text/csv")}
        )
        self.assertEqual(resp.status_code, 200)
        b_res = resp.json()
        self.assertEqual(b_res["summary"]["total_dies"], 5)
        self.assertEqual(len(b_res["results"]), 5)

        # 2. Invalid CSV Schema (Missing parametric features)
        inv_buffer = io.BytesIO(b"wafer_id,die_row,die_col\nW_1,0,0\n")
        resp_inv = self.client.post(
            "/api/v1/inference/batch",
            files={"file": ("invalid.csv", inv_buffer, "text/csv")}
        )
        self.assertEqual(resp_inv.status_code, 422)
        err = resp_inv.json()
        self.assertEqual(err["detail"]["code"], "INVALID_CSV_SCHEMA")

    def test_11_block_attribution_endpoint(self):
        """GET /api/v1/inference/block-attribution returns 2,000 waveform readings and attributions."""
        resp = self.client.get("/api/v1/inference/block-attribution?wafer_id=W_N_0057&die_row=32&die_col=23")
        self.assertEqual(resp.status_code, 200)
        attr_data = resp.json()
        self.assertEqual(attr_data["block_count"], 2000)
        self.assertEqual(len(attr_data["readings"]), 2000)
        self.assertEqual(len(attr_data["attributions"]), 2000)
        self.assertGreaterEqual(attr_data["peak_attribution_value"], 0.0)

    def test_12_evaluation_run_endpoint(self):
        """POST /api/v1/evaluation/run executes live full evaluation on all 27,972 dies."""
        resp = self.client.post("/api/v1/evaluation/run")
        self.assertEqual(resp.status_code, 200)
        eval_data = resp.json()
        self.assertEqual(eval_data["status"], "COMPLETED")
        self.assertEqual(eval_data["validation_dies_evaluated"], 27972)
        self.assertEqual(eval_data["integrity_checks"]["prediction_mismatch_count"], 0)
        self.assertEqual(eval_data["integrity_checks"]["delta_f1_vs_authoritative"], 0.0)

    def test_13_ml_regression_equivalence(self):
        """Regression Verification: API responses match direct Python ML engine within 1e-5."""
        sample_dies = [
            {"wafer_id": "W_N_0057", "die_row": 32, "die_col": 23}, # Recovered FN
            {"wafer_id": "W_N_0001", "die_row": 0, "die_col": 13},  # Fast Path Pass
            {"wafer_id": "W_N_0057", "die_row": 18, "die_col": 26}  # Recovered FN
        ]

        for s in sample_dies:
            # 1. Direct Python invocation
            match = state.diagnostics_df[
                (state.diagnostics_df["wafer_id"] == s["wafer_id"]) &
                (state.diagnostics_df["die_row"] == s["die_row"]) &
                (state.diagnostics_df["die_col"] == s["die_col"])
            ].iloc[0].to_dict()
            
            raw_res = predict_single_die_live(match, state.model_A, state.model_B)

            # 2. FastAPI endpoint invocation
            api_resp = self.client.post("/api/v1/inference/die", json=s)
            self.assertEqual(api_resp.status_code, 200)
            api_res = api_resp.json()

            # Compare exact numerical values
            self.assertAlmostEqual(api_res["prob_Model_A"], float(raw_res["prob_Model_A"]), places=4)
            self.assertEqual(api_res["decision_Model_A"], int(raw_res["decision_Model_A"]))
            self.assertEqual(api_res["in_gate"], bool(raw_res["in_gate"]))
            self.assertEqual(api_res["final_decision"], int(raw_res["final_decision"]))
            self.assertAlmostEqual(api_res["final_probability"], float(raw_res["final_probability"]), places=4)
            if raw_res["prob_Model_B"] is not None:
                self.assertAlmostEqual(api_res["prob_Model_B"], float(raw_res["prob_Model_B"]), places=4)
            self.assertEqual(api_res["compute_path"], str(raw_res["compute_path"]))

if __name__ == "__main__":
    unittest.main()
