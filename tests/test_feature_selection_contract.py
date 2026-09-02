from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tools/feature_selection_pipeline.py"
SPEC = importlib.util.spec_from_file_location("feature_selection_pipeline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FeatureSelectionContractTest(unittest.TestCase):
    def test_main_sequence_feature_dimensions(self) -> None:
        sequences = [MODULE.WT_141]
        self.assertEqual(MODULE.build_mut(sequences)[0].shape, (1, 52))
        self.assertEqual(MODULE.build_wd(sequences)[0].shape, (1, 208))
        self.assertEqual(MODULE.build_struct(sequences)[0].shape, (1, 43))

    def test_mutation_features_are_force_kept(self) -> None:
        sequences = [MODULE.WT_141]
        names = []
        for builder in (MODULE.build_mut, MODULE.build_wd, MODULE.build_struct):
            _, block_names = builder(sequences)
            names.extend(block_names)
        mask = MODULE.make_force_keep_mask(MODULE.np.array(names), "mut")
        self.assertEqual(int(mask.sum()), 52)
        self.assertTrue(all(name.startswith("mut_") for name in MODULE.np.array(names)[mask]))

    def test_formal_config_has_control_and_promotion_thresholds(self) -> None:
        config = json.loads(
            (ROOT / "configs/experiments/feature_selection_30seed.json").read_text(
                encoding="utf-8"
            )
        )
        methods = config["arguments"]["methods"].split(",")
        self.assertIn("all", methods)
        self.assertEqual(config["arguments"]["force-keep-groups"], "mut")
        self.assertEqual(config["arguments"]["seeds"], 30)
        self.assertEqual(config["acceptance"]["promotion_mean_delta_min"], 0.005)
        self.assertEqual(config["acceptance"]["promotion_win_rate_min"], 0.6)


if __name__ == "__main__":
    unittest.main()
