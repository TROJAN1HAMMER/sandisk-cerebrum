import os
import json
import pickle
import unittest
import numpy as np
import pandas as pd
import torch

from src.model_a import CandidateModelAWrapper
from src.model_b import CandidateModelBWrapper, Block1DCNN
from src.cascade import DualStreamCandidateCascade
from src.live_inference import predict_single_die_live

class TestB3ProductionArchitectureAndRollback(unittest.TestCase):
    """
    Authoritative test suite validating the promoted B3 Multimodal Dual-Stream Cascade
    production architecture, feature contracts, selective inference, metadata,
    and B2 rollback preservation.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.models_dir = os.path.join(cls.base_dir, 'models')
        cls.candidates_dir = os.path.join(cls.models_dir, 'candidates')
        cls.release_json_path = os.path.join(cls.models_dir, 'model_release.json')
        cls.rollback_dir = os.path.join(cls.models_dir, 'b2_rollback')

        with open(cls.release_json_path, 'r') as f:
            cls.release_meta = json.load(f)

    def test_01_b3_model_a_loads(self):
        """1. Verify B3 Model A artifact loads successfully and has expected type."""
        model_a_path = os.path.join(self.candidates_dir, 'model_A_enhanced.pkl')
        self.assertTrue(os.path.exists(model_a_path), "B3 Model A artifact must exist")
        with open(model_a_path, 'rb') as f:
            mA = pickle.load(f)
        self.assertEqual(type(mA).__name__, "LGBMClassifier")
        wrapper_A = CandidateModelAWrapper()
        self.assertEqual(wrapper_A.n_features, 517)

    def test_02_b3_model_b_loads(self):
        """2. Verify B3 Model B artifact loads successfully and has expected type."""
        model_b_path = os.path.join(self.candidates_dir, 'model_B_tab_enhanced.pkl')
        self.assertTrue(os.path.exists(model_b_path), "B3 Model B artifact must exist")
        with open(model_b_path, 'rb') as f:
            mB = pickle.load(f)
        self.assertEqual(type(mB).__name__, "LGBMClassifier")
        wrapper_B = CandidateModelBWrapper()
        self.assertEqual(wrapper_B.n_tabular_features, 528)

    def test_03_cnn_loads(self):
        """3. Verify CNN architecture loads successfully with 41,153 parameters."""
        cnn_path = os.path.join(self.models_dir, 'cnn_1d_block_model.pt')
        self.assertTrue(os.path.exists(cnn_path), "CNN artifact must exist")
        cnn = Block1DCNN()
        cnn.load_state_dict(torch.load(cnn_path, map_location='cpu'))
        cnn.eval()
        param_count = sum(p.numel() for p in cnn.parameters())
        self.assertEqual(param_count, 41153)

    def test_04_pca_loads(self):
        """4. Verify PCA pipeline loads successfully with 25 components."""
        pca_path = os.path.join(self.candidates_dir, 'pca_pipeline.pkl')
        self.assertTrue(os.path.exists(pca_path), "PCA pipeline must exist")
        with open(pca_path, 'rb') as f:
            pca_bundle = pickle.load(f)
        self.assertIn('pca', pca_bundle)
        self.assertIn('pass_mean', pca_bundle)
        self.assertIn('pass_std', pca_bundle)
        self.assertEqual(pca_bundle['pca'].n_components_, 25)
        self.assertEqual(pca_bundle['pass_mean'].shape, (500,))
        self.assertEqual(pca_bundle['pass_std'].shape, (500,))

    def test_05_model_a_517_feature_contract(self):
        """5. Verify Model A feature contract requires exactly 517 features."""
        with open(os.path.join(self.candidates_dir, 'model_A_enhanced.pkl'), 'rb') as f:
            mA = pickle.load(f)
        self.assertEqual(len(mA.feature_name_), 517)
        self.assertEqual(self.release_meta["active_configuration"]["model_A"]["feature_count"], 517)

    def test_06_model_b_528_tabular_feature_contract(self):
        """6. Verify Model B tabular feature contract requires exactly 528 features."""
        with open(os.path.join(self.candidates_dir, 'model_B_tab_enhanced.pkl'), 'rb') as f:
            mB = pickle.load(f)
        self.assertEqual(len(mB.feature_name_), 528)
        self.assertEqual(self.release_meta["active_configuration"]["model_B"]["feature_count"], 528)

    def test_07_cnn_2000_block_contract(self):
        """7. Verify CNN 2,000 raw block contract and forward pass."""
        cnn = Block1DCNN()
        cnn.load_state_dict(torch.load(os.path.join(self.models_dir, 'cnn_1d_block_model.pt'), map_location='cpu'))
        cnn.eval()
        dummy_in = torch.randn(2, 1, 2000)
        out = cnn(dummy_in)
        self.assertEqual(out.shape, torch.Size([2]))

    def test_08_b3_gate_window(self):
        """8. Verify B3 standard production gate is [0.250, 0.550]."""
        gate = self.release_meta["active_configuration"]["cascade_gate"]
        self.assertEqual(gate["lower"], 0.250)
        self.assertEqual(gate["upper"], 0.550)

    def test_09_tau_a_threshold(self):
        """9. Verify tau_A threshold is 0.5100."""
        tau_A = self.release_meta["active_configuration"]["model_A"]["threshold"]
        self.assertEqual(tau_A, 0.5100)

    def test_10_tau_b_threshold(self):
        """10. Verify tau_B threshold is 0.5200."""
        tau_B = self.release_meta["active_configuration"]["model_B"]["threshold"]
        self.assertEqual(tau_B, 0.5200)

    def test_11_fusion_formula(self):
        """11. Verify late probability fusion weights are 0.90 tabular and 0.10 CNN."""
        fusion = self.release_meta["active_configuration"]["fusion"]
        self.assertEqual(fusion["tabular_weight"], 0.90)
        self.assertEqual(fusion["cnn_weight"], 0.10)
        # Mathematical verification
        p_tab = 0.60
        p_cnn = 0.40
        p_fused = 0.90 * p_tab + 0.10 * p_cnn
        self.assertAlmostEqual(p_fused, 0.58, places=4)

    def test_12_cnn_not_invoked_on_fast_path(self):
        """12. Verify CNN is NOT evaluated on fast-path dies."""
        cand_A = CandidateModelAWrapper()
        cand_B = CandidateModelBWrapper()
        
        # Test low probability die (p_A = 0.10 < 0.250)
        die_pass = {
            'wafer_id': 'W_TEST', 'die_row': 5, 'die_col': 5,
            'prob_Model_A': 0.10,
            'block_readings': ' '.join(['100.0'] * 2000)
        }
        res_pass = predict_single_die_live(die_pass, cand_A, cand_B, gate_lower=0.250, gate_upper=0.550)
        self.assertEqual(res_pass['compute_path'], 'FAST PATH')
        self.assertFalse(res_pass['model_B_invoked'])
        self.assertIsNone(res_pass['prob_Model_B'])
        self.assertEqual(res_pass['final_probability'], 0.10)
        self.assertEqual(res_pass['final_decision'], 0)

        # Test high probability die (p_A = 0.80 > 0.550)
        die_fail = {
            'wafer_id': 'W_TEST', 'die_row': 10, 'die_col': 10,
            'prob_Model_A': 0.80,
            'block_readings': ' '.join(['100.0'] * 2000)
        }
        res_fail = predict_single_die_live(die_fail, cand_A, cand_B, gate_lower=0.250, gate_upper=0.550)
        self.assertEqual(res_fail['compute_path'], 'FAST PATH')
        self.assertFalse(res_fail['model_B_invoked'])
        self.assertIsNone(res_fail['prob_Model_B'])
        self.assertEqual(res_fail['final_probability'], 0.80)
        self.assertEqual(res_fail['final_decision'], 1)

    def test_13_standard_b3_policy_selected(self):
        """13. Verify standard B3 cascade policy is used as production default."""
        cascade = DualStreamCandidateCascade()
        cascade.set_policy("candidate_standard")
        self.assertEqual(cascade.lower_gate, 0.250)
        self.assertEqual(cascade.upper_gate, 0.550)
        self.assertEqual(cascade.tau_A, 0.5100)
        self.assertEqual(cascade.tau_B, 0.5200)

    def test_14_experimental_high_recall_not_selected(self):
        """14. Verify experimental high-recall policy [0.200, 0.500] is NOT production."""
        active_gate = self.release_meta["active_configuration"]["cascade_gate"]
        self.assertNotEqual(active_gate["lower"], 0.200)
        self.assertNotEqual(active_gate["upper"], 0.500)
        self.assertNotEqual(active_gate["tau_B"], 0.5000)
        self.assertEqual(active_gate["experimental_policy_status"], "REJECTED_FOR_PRODUCTION")

    def test_15_production_release_metadata_identifies_b3(self):
        """15. Verify production release metadata identifies B3 and status PRODUCTION."""
        self.assertEqual(self.release_meta["model_version"], "B3")
        self.assertEqual(self.release_meta["model_status"], "PRODUCTION")
        self.assertIn("Multimodal Dual-Stream", self.release_meta["model_family"])

        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.deps import init_app_state
        init_app_state()
        client = TestClient(app)
        resp = client.get("/api/v1/models/release")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_version"], "B3")
        self.assertEqual(data["model_status"], "PRODUCTION")

    def test_16_b2_rollback_artifacts_preserved(self):
        """16. Verify immutable B2 rollback artifacts remain available and intact."""
        self.assertTrue(os.path.exists(self.rollback_dir), "b2_rollback directory must exist")
        self.assertTrue(os.path.exists(os.path.join(self.rollback_dir, 'model_A.pkl')), "B2 Model A must exist in rollback")
        self.assertTrue(os.path.exists(os.path.join(self.rollback_dir, 'model_B.pkl')), "B2 Model B must exist in rollback")
        self.assertTrue(os.path.exists(os.path.join(self.rollback_dir, 'model_release.json')), "B2 release metadata must exist in rollback")
        
        # Verify historical backup in models/
        backup_b = os.path.join(self.models_dir, 'model_B_original_production_backup.pkl')
        self.assertTrue(os.path.exists(backup_b), "Original B0 backup must exist")
        
        # Verify evaluations key preserves historical B2 metrics
        self.assertIn("historical_b2_production", self.release_meta["evaluations"])
        b2_eval = self.release_meta["evaluations"]["historical_b2_production"]["cascade_metrics"]
        self.assertEqual(b2_eval["f1"], 0.5177)
        self.assertEqual(b2_eval["precision"], 0.8522)
        self.assertEqual(b2_eval["recall"], 0.3717)

if __name__ == '__main__':
    unittest.main()
