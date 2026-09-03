import os
import json
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)

class TestPhase3CMetricsConsistency(unittest.TestCase):

    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.preds_csv = os.path.join(self.base_dir, 'results', 'clean_model_B_predictions.csv')
        self.auth_json = os.path.join(self.base_dir, 'results', 'AUTHORITATIVE_CLEAN_METRICS.json')
        self.bench_csv = os.path.join(self.base_dir, 'results', 'final_clean_benchmark.csv')
        
        self.df = pd.read_csv(self.preds_csv)
        with open(self.auth_json, 'r') as f:
            self.auth = json.load(f)
            
        self.bench = pd.read_csv(self.bench_csv)
        
        self.y_true = self.df['actual_label'].values
        self.p_A = self.df['prob_Model_A'].values
        self.p_B = self.df['prob_Model_B'].values
        self.pred_A = self.df['decision_Model_A'].values
        self.pred_B = self.df['decision_Model_B'].values

    def test_sample_counts_and_balance(self):
        """Verify total rows, positive count, negative count, and confusion matrix balances."""
        N_total = len(self.df)
        self.assertEqual(N_total, 27972)
        self.assertEqual(int(self.y_true.sum()), 1056)
        self.assertEqual(int((1 - self.y_true).sum()), 26916)
        
        for key in ['model_A', 'model_B', 'cascade_gate_B', 'cascade_gate_C']:
            cm = self.auth[key]['confusion_matrix']
            total = cm['tp'] + cm['fn'] + cm['fp'] + cm['tn']
            self.assertEqual(total, N_total, f"Confusion matrix for {key} does not sum to 27,972")
            self.assertEqual(cm['tp'] + cm['fn'], 1056, f"Positives in {key} do not sum to 1,056")
            self.assertEqual(cm['fp'] + cm['tn'], 26916, f"Negatives in {key} do not sum to 26,916")

    def test_model_A_metrics(self):
        """Verify Model A metrics match raw recalculation."""
        mA = self.auth['model_A']
        self.assertEqual(mA['confusion_matrix']['tp'], 357)
        self.assertEqual(mA['confusion_matrix']['fn'], 699)
        self.assertEqual(mA['confusion_matrix']['fp'], 42)
        self.assertEqual(mA['confusion_matrix']['tn'], 26874)
        
        f1 = f1_score(self.y_true, self.pred_A)
        self.assertAlmostEqual(f1, mA['f1'], places=4)
        
        p, r, _ = precision_recall_curve(self.y_true, self.p_A)
        auc_pr = auc(r, p)
        self.assertAlmostEqual(auc_pr, mA['auc_pr'], places=3)

    def test_model_B_metrics(self):
        """Verify Model B metrics match raw recalculation."""
        mB = self.auth['model_B']
        self.assertEqual(mB['confusion_matrix']['tp'], 365)
        self.assertEqual(mB['confusion_matrix']['fn'], 691)
        self.assertEqual(mB['confusion_matrix']['fp'], 130)
        self.assertEqual(mB['confusion_matrix']['tn'], 26786)
        
        f1 = f1_score(self.y_true, self.pred_B)
        self.assertAlmostEqual(f1, mB['f1'], places=4)
        
        p, r, _ = precision_recall_curve(self.y_true, self.p_B)
        auc_pr = auc(r, p)
        self.assertAlmostEqual(auc_pr, mB['auc_pr'], places=4)

    def test_cascade_gate_B_consistency(self):
        """Verify Cascade Gate B [0.25, 0.55] logic, escalation count, and recovery math."""
        gB = self.auth['cascade_gate_B']
        mask = (self.p_A >= 0.25) & (self.p_A <= 0.55)
        self.assertEqual(int(mask.sum()), gB['escalation_count'])
        self.assertEqual(gB['escalation_count'], 3317)
        
        pred_final = np.where(mask, (self.p_B >= 0.5376).astype(int), self.pred_A)
        tp, fn, fp, tn = (
            int(((self.y_true == 1) & (pred_final == 1)).sum()),
            int(((self.y_true == 1) & (pred_final == 0)).sum()),
            int(((self.y_true == 0) & (pred_final == 1)).sum()),
            int(((self.y_true == 0) & (pred_final == 0)).sum())
        )
        self.assertEqual(tp, gB['confusion_matrix']['tp'])
        self.assertEqual(fn, gB['confusion_matrix']['fn'])
        self.assertEqual(fp, gB['confusion_matrix']['fp'])
        self.assertEqual(tn, gB['confusion_matrix']['tn'])
        
        # FN Recovery Breakdown
        fn_a_mask = (self.y_true == 1) & (self.pred_A == 0)
        tp_a_mask = (self.y_true == 1) & (self.pred_A == 1)
        gross_rec = int((pred_final[fn_a_mask] == 1).sum())
        tp_lost = int((pred_final[tp_a_mask] == 0).sum())
        net_gain = tp - 357
        
        self.assertEqual(gross_rec, gB['fn_recovery_metrics']['gross_fn_recovered'])
        self.assertEqual(gross_rec, 37)
        self.assertEqual(tp_lost, 5)
        self.assertEqual(net_gain, 32)
        self.assertEqual(gross_rec - tp_lost, net_gain)

    def test_cascade_gate_C_consistency(self):
        """Verify Cascade Gate C [0.30, 0.60] logic, escalation count, and recovery math."""
        gC = self.auth['cascade_gate_C']
        mask = (self.p_A >= 0.30) & (self.p_A <= 0.60)
        self.assertEqual(int(mask.sum()), gC['escalation_count'])
        self.assertEqual(gC['escalation_count'], 1626)
        
        pred_final = np.where(mask, (self.p_B >= 0.5376).astype(int), self.pred_A)
        tp, fn, fp, tn = (
            int(((self.y_true == 1) & (pred_final == 1)).sum()),
            int(((self.y_true == 1) & (pred_final == 0)).sum()),
            int(((self.y_true == 0) & (pred_final == 1)).sum()),
            int(((self.y_true == 0) & (pred_final == 0)).sum())
        )
        self.assertEqual(tp, gC['confusion_matrix']['tp'])
        self.assertEqual(fn, gC['confusion_matrix']['fn'])
        self.assertEqual(fp, gC['confusion_matrix']['fp'])
        self.assertEqual(tn, gC['confusion_matrix']['tn'])
        
        fn_a_mask = (self.y_true == 1) & (self.pred_A == 0)
        tp_a_mask = (self.y_true == 1) & (self.pred_A == 1)
        gross_rec = int((pred_final[fn_a_mask] == 1).sum())
        tp_lost = int((pred_final[tp_a_mask] == 0).sum())
        net_gain = tp - 357
        
        self.assertEqual(gross_rec, gC['fn_recovery_metrics']['gross_fn_recovered'])
        self.assertEqual(gross_rec, 30)
        self.assertEqual(tp_lost, 10)
        self.assertEqual(net_gain, 20)
        self.assertEqual(gross_rec - tp_lost, net_gain)

if __name__ == '__main__':
    unittest.main()
