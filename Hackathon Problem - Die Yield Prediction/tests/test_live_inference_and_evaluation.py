import os
import json
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix

from src.model_a import ModelAWrapper
from src.live_inference import (
    predict_single_die_live,
    predict_batch_df_live,
    validate_csv_schema,
    execute_full_live_evaluation,
    MODEL_A_THRESHOLD,
    MODEL_B_THRESHOLD,
    GATE_B_LOWER,
    GATE_B_UPPER
)
import pickle

class TestLiveInferenceAndEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.models_dir = os.path.join(cls.base_dir, 'models')
        cls.results_dir = os.path.join(cls.base_dir, 'results')
        
        cls.model_A = ModelAWrapper(model_dir=cls.models_dir)
        with open(os.path.join(cls.models_dir, 'model_B.pkl'), 'rb') as f:
            cls.model_B = pickle.load(f)
            
        cls.diag_df = pd.read_csv(os.path.join(cls.results_dir, 'val_die_diagnostics.csv'))
        with open(os.path.join(cls.results_dir, 'AUTHORITATIVE_CLEAN_METRICS.json'), 'r') as f:
            cls.auth_metrics = json.load(f)

    def test_csv_schema_validation(self):
        """Test schema validator with valid and invalid dataframes."""
        # 1. Valid dataframe
        valid_cols = ['wafer_id', 'die_row', 'die_col'] + [f'feature_{i}' for i in range(1, 501)]
        df_valid = pd.DataFrame(np.zeros((3, len(valid_cols))), columns=valid_cols)
        df_valid['wafer_id'] = 'W_TEST_0001'
        val_res = validate_csv_schema(df_valid)
        self.assertTrue(val_res['valid'])
        self.assertEqual(len(val_res['errors']), 0)
        self.assertEqual(val_res['num_rows'], 3)
        
        # 2. Missing coordinate columns
        df_invalid = pd.DataFrame(np.zeros((2, 500)), columns=[f'feature_{i}' for i in range(1, 501)])
        val_res_inv = validate_csv_schema(df_invalid)
        self.assertFalse(val_res_inv['valid'])
        self.assertTrue(any('wafer_id' in e for e in val_res_inv['errors']))
        
        # 3. Missing parametric features
        df_incomplete = pd.DataFrame({'wafer_id': ['W_1'], 'die_row': [10], 'die_col': [10], 'feature_1': [0.5]})
        val_res_inc = validate_csv_schema(df_incomplete)
        self.assertFalse(val_res_inc['valid'])
        self.assertTrue(any('parametric' in e.lower() for e in val_res_inc['errors']))

    def test_live_single_die_inference_fast_path(self):
        """Test confident passing die that bypasses Model B (Fast Path)."""
        demo_3 = self.diag_df.iloc[0].to_dict()
        res = predict_single_die_live(demo_3, self.model_A, self.model_B)
        
        self.assertFalse(res['in_gate'])
        self.assertEqual(res['compute_path'], "FAST PATH")
        self.assertFalse(res['model_B_invoked'])
        self.assertIsNone(res['prob_Model_B'])
        self.assertEqual(res['final_decision'], 0)
        self.assertEqual(res['decision_Model_A'], 0)

    def test_live_single_die_inference_escalated_path(self):
        """Test ambiguous die that is escalated to Model B and recovered (Escalated Path)."""
        demo_1 = self.diag_df.iloc[3736].to_dict() # W_N_0057 R:32 C:23
        res = predict_single_die_live(demo_1, self.model_A, self.model_B)
        
        self.assertTrue(res['in_gate'])
        self.assertEqual(res['compute_path'], "ESCALATED PATH")
        self.assertTrue(res['model_B_invoked'])
        self.assertIsNotNone(res['prob_Model_B'])
        self.assertGreaterEqual(res['prob_Model_B'], 0.5376)
        self.assertEqual(res['final_decision'], 1)

    def test_batch_live_prediction_output(self):
        """Test batch prediction function and summary statistics."""
        batch_sample = self.diag_df.head(20).copy()
        batch_res = predict_batch_df_live(batch_sample, self.model_A, self.model_B)
        
        self.assertIn('results_df', batch_res)
        self.assertIn('summary', batch_res)
        
        summary = batch_res['summary']
        self.assertEqual(summary['total_dies'], 20)
        self.assertEqual(summary['predicted_fails'] + summary['predicted_passes'], 20)
        self.assertAlmostEqual(summary['escalation_rate_pct'] + summary['compute_saved_pct'], 100.0, places=2)
        
        res_df = batch_res['results_df']
        self.assertEqual(len(res_df), 20)
        self.assertIn('final_probability', res_df.columns)
        self.assertIn('final_decision', res_df.columns)
        self.assertIn('compute_path', res_df.columns)

    def test_full_live_evaluation_metrics_and_integrity(self):
        """Test that full dynamic evaluation reproduces authoritative clean metrics."""
        eval_res = execute_full_live_evaluation(self.model_A, self.model_B, self.diag_df, self.auth_metrics)
        
        # Verify Model A
        self.assertEqual(eval_res['model_A']['tp'], 357)
        self.assertEqual(eval_res['model_A']['fn'], 699)
        self.assertEqual(eval_res['model_A']['fp'], 42)
        self.assertEqual(eval_res['model_A']['tn'], 26874)
        self.assertAlmostEqual(eval_res['model_A']['f1'], 0.4907, places=4)
        self.assertAlmostEqual(eval_res['model_A']['auc_pr'], 0.4498, places=4)
        
        # Verify Gate B
        gb = eval_res['cascade_gate_B']
        self.assertEqual(gb['tp'], 389)
        self.assertEqual(gb['fn'], 667)
        self.assertEqual(gb['fp'], 89)
        self.assertEqual(gb['tn'], 26827)
        self.assertEqual(gb['escalation_count'], 3317)
        self.assertEqual(gb['gross_fn_recovered'], 37)
        self.assertEqual(gb['tp_lost'], 5)
        self.assertEqual(gb['net_tp_gain'], 32)
        self.assertEqual(gb['new_fp'], 47)
        self.assertAlmostEqual(gb['f1'], 0.5072, places=4)
        self.assertAlmostEqual(gb['auc_pr'], 0.4677, places=4)
        
        # Verify Integrity Checks
        integ = eval_res['integrity_checks']
        self.assertEqual(integ['canonical_split'], "PASS")
        self.assertEqual(integ['runtime_reproducible'], "PASS")
        self.assertEqual(integ['prediction_mismatch_count'], 0)
        self.assertEqual(integ['delta_f1_vs_authoritative'], 0.0)
        self.assertEqual(integ['delta_auc_vs_authoritative'], 0.0)

if __name__ == '__main__':
    unittest.main()
