import os
import json
import pickle
import unittest

class TestModelReleaseB2(unittest.TestCase):
    """
    Dedicated test suite validating the final B2 release metadata,
    hyperparameters, gate boundaries, and artifact dimensions.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.models_dir = os.path.join(cls.base_dir, 'models')
        cls.release_json_path = os.path.join(cls.models_dir, 'model_release.json')

        with open(cls.release_json_path, 'r') as f:
            cls.release_meta = json.load(f)

        with open(os.path.join(cls.models_dir, 'model_B.pkl'), 'rb') as f:
            cls.model_B = pickle.load(f)

        backup_path = os.path.join(cls.models_dir, 'model_B_original_production_backup.pkl')
        cls.has_backup = os.path.exists(backup_path)
        if cls.has_backup:
            with open(backup_path, 'rb') as f:
                cls.model_B_backup = pickle.load(f)

    def test_01_final_model_version(self):
        """Verify model release version is B2 and status is final validated candidate."""
        self.assertEqual(self.release_meta["model_version"], "B2")
        self.assertEqual(self.release_meta["model_status"], "FINAL VALIDATED CANDIDATE")

    def test_02_gate_boundaries(self):
        """Verify production gate boundaries are [0.295, 0.550]."""
        gate = self.release_meta["active_configuration"]["cascade_gate"]
        self.assertEqual(gate["lower"], 0.295)
        self.assertEqual(gate["upper"], 0.550)

    def test_03_decision_thresholds(self):
        """Verify decision thresholds tau_A = 0.5100 and tau_B = 0.5376."""
        tau_A = self.release_meta["active_configuration"]["model_A"]["threshold"]
        tau_B = self.release_meta["active_configuration"]["model_B"]["threshold"]
        self.assertEqual(tau_A, 0.5100)
        self.assertEqual(tau_B, 0.5376)

    def test_04_b2_feature_count(self):
        """Verify B2 feature count is 536 and model artifact matches."""
        feat_count = self.release_meta["active_configuration"]["model_B"]["feature_count"]
        self.assertEqual(feat_count, 536)
        self.assertEqual(self.model_B.n_features_in_, 536)

    def test_05_production_model_backup_preserved(self):
        """Verify immutable archival backup of original B0 production model exists with 526 features."""
        self.assertTrue(self.has_backup, "Historical B0 model backup must exist")
        self.assertEqual(self.model_B_backup.n_features_in_, 526)

    def test_06_model_metadata_consistency(self):
        """Verify independent validation numbers in release metadata match EXP-04 benchmark."""
        ind_eval = self.release_meta["evaluations"]["final_independent_validation"]
        self.assertEqual(ind_eval["wafers_count"], 40)
        self.assertEqual(ind_eval["dies_count"], 32598)
        self.assertEqual(ind_eval["actual_defects"], 1380)
        self.assertEqual(ind_eval["cascade_metrics"]["f1"], 0.5177)
        self.assertEqual(ind_eval["cascade_metrics"]["pr_auc"], 0.4808)
        self.assertEqual(ind_eval["cascade_metrics"]["roc_auc"], 0.8078)
        self.assertEqual(ind_eval["cascade_metrics"]["precision"], 0.8522)
        self.assertEqual(ind_eval["cascade_metrics"]["recall"], 0.3717)
        self.assertEqual(ind_eval["cascade_metrics"]["compute_saved_pct"], 93.78)
        self.assertEqual(ind_eval["cascade_metrics"]["tp"], 513)

    def test_07_model_release_api_endpoint(self):
        """Verify GET /api/v1/models/release returns 200 with B2 release metadata."""
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.deps import init_app_state
        init_app_state()
        client = TestClient(app)
        resp = client.get("/api/v1/models/release")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_version"], "B2")
        self.assertEqual(data["model_status"], "FINAL VALIDATED CANDIDATE")
        self.assertEqual(data["active_configuration"]["cascade_gate"]["lower"], 0.295)
        self.assertEqual(data["active_configuration"]["cascade_gate"]["upper"], 0.550)

    def test_08_candidate_models_isolation_and_api(self):
        """Verify Candidate B3 models exist in models/candidates/ and /candidates endpoint returns 200."""
        cand_dir = os.path.join(self.models_dir, 'candidates')
        self.assertTrue(os.path.exists(os.path.join(cand_dir, 'model_A_enhanced.pkl')), "Candidate Model A must exist")
        self.assertTrue(os.path.exists(os.path.join(cand_dir, 'model_B_tab_enhanced.pkl')), "Candidate Model B must exist")
        self.assertTrue(os.path.exists(os.path.join(cand_dir, 'candidate_metadata.json')), "Candidate metadata must exist")

        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        resp = client.get("/api/v1/models/candidates")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["active_production_release"], "B2")
        self.assertEqual(data["candidate_model_version"], "B3-DualStream")
        self.assertEqual(data["status"], "VALIDATED_EXPERIMENTAL_CANDIDATE")
        self.assertEqual(data["features"]["model_A_features"], 517)
        self.assertEqual(data["features"]["model_B_tabular_features"], 528)
        self.assertIn("evaluation_summary", data)

if __name__ == '__main__':
    unittest.main()
