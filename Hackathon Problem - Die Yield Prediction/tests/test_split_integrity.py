import os
import json
import unittest
import pandas as pd
import numpy as np

from src.data_loader import load_canonical_wafer_split, get_wafer_split, load_eligible_data
from src.evaluation_protocol import (
    validate_eligible_dataset, validate_features_not_leaked,
    validate_wafer_partition_integrity, calculate_evaluation_metrics
)

class TestSplitIntegrity(unittest.TestCase):

    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.split_json = os.path.join(self.base_dir, 'results', 'canonical_wafer_split.json')
        self.clean_val_csv = os.path.join(self.base_dir, 'results', 'val_predictions_model_A.csv')

    def test_canonical_split_loads_and_counts(self):
        """Verify that canonical split loads successfully and matches 128/32 exactly."""
        train_wafers, val_wafers, meta = load_canonical_wafer_split(self.split_json)
        self.assertEqual(len(train_wafers), 128, "Expected exactly 128 train wafers")
        self.assertEqual(len(val_wafers), 32, "Expected exactly 32 validation wafers")
        self.assertEqual(len(train_wafers.intersection(val_wafers)), 0, "Train and val sets must have 0 overlap")

    def test_split_determinism(self):
        """Verify that multiple loads return identical sets."""
        tr1, val1, _ = load_canonical_wafer_split(self.split_json)
        tr2, val2, _ = load_canonical_wafer_split(self.split_json)
        self.assertEqual(tr1, tr2)
        self.assertEqual(val1, val2)

    def test_clean_model_a_predictions_integrity(self):
        """Verify that clean Model A validation predictions file contains strictly canonical val wafers."""
        self.assertTrue(os.path.exists(self.clean_val_csv), f"Missing {self.clean_val_csv}")
        df = pd.read_csv(self.clean_val_csv)
        train_wafers, val_wafers, _ = load_canonical_wafer_split(self.split_json)
        
        val_csv_wafers = set(df['wafer_id'].unique())
        self.assertEqual(val_csv_wafers, val_wafers, "Validation prediction wafers must match canonical val wafers")
        self.assertEqual(len(val_csv_wafers.intersection(train_wafers)), 0, "No training wafer must exist in validation predictions")
        self.assertEqual(len(df), 27972, "Expected 27,972 rows in clean Model A validation predictions")
        
        # Verify metrics
        y_true = df['label'].values
        y_prob = df['prob'].values
        metrics = calculate_evaluation_metrics(y_true, y_prob, threshold=0.51)
        self.assertAlmostEqual(metrics['f1'], 0.4900, places=3)
        self.assertAlmostEqual(metrics['auc_pr'], 0.4493, places=3)
        self.assertAlmostEqual(metrics['precision'], 0.8903, places=3)
        self.assertAlmostEqual(metrics['recall'], 0.3381, places=3)
        self.assertEqual(metrics['confusion_matrix']['tp'], 357)
        self.assertEqual(metrics['confusion_matrix']['fn'], 699)
        self.assertEqual(metrics['confusion_matrix']['fp'], 44)
        self.assertEqual(metrics['confusion_matrix']['tn'], 26872)

    def test_feature_leakage_assertion(self):
        """Verify that forbidden columns raise exceptions when passed to feature validator."""
        valid_cols = [f'feature_{i}' for i in range(1, 10)] + ['radial_dist', 'density_3x3']
        self.assertTrue(validate_features_not_leaked(valid_cols))
        
        for bad_col in ['wafer_id', 'old_label', 'label', 'block_readings']:
            with self.assertRaises(ValueError):
                validate_features_not_leaked(valid_cols + [bad_col])

if __name__ == '__main__':
    unittest.main()
