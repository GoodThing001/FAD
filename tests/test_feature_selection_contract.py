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

    def test_all_safe_feature_matrix_dimensions_and_alignment(self) -> None:
        data_dir = ROOT / "data/proxy2000_v2"
        full_df = MODULE.pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
        sequence_blocks = [
            builder(full_df["Sequence"].values)[0]
            for builder in (MODULE.build_mut, MODULE.build_wd, MODULE.build_struct)
        ]
        nupack, names = MODULE.load_nupack_features(data_dir, full_df, blocks="all")
        self.assertEqual(nupack.shape, (2000, 1192))
        self.assertEqual(len(names), 1192)
        self.assertTrue(
            all(name.startswith(("nupack_", "delta_nupack_")) for name in names)
        )
        self.assertEqual(sum(block.shape[1] for block in sequence_blocks) + nupack.shape[1], 1495)

    def test_useful_nupack_blocks_are_exactly_192d(self) -> None:
        data_dir = ROOT / "data/proxy2000_v2"
        full_df = MODULE.pd.read_csv(data_dir / "fad_proxy2000_v2_full.csv")
        nupack, names = MODULE.load_nupack_features(data_dir, full_df, blocks="useful")
        self.assertEqual(nupack.shape, (2000, 192))
        self.assertEqual(len(names), 192)
        self.assertTrue(
            all(name.startswith("nupack_") and not name.startswith("delta_")
                for name in names)
        )

    def test_all_safe_formal_config(self) -> None:
        config = json.loads(
            (ROOT / "configs/experiments/feature_selection_all_safe_30seed.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(config["arguments"]["with-nupack"])
        self.assertEqual(config["arguments"]["nupack-blocks"], "all")
        self.assertEqual(config["arguments"]["expected-features"], 1495)
        self.assertEqual(config["acceptance"]["expected_input_features"], 1495)
        self.assertEqual(config["acceptance"]["expected_rows"], 360)

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
        expected_rows = (
            config["arguments"]["seeds"]
            * len(config["arguments"]["methods"].split(","))
            * len(config["arguments"]["eval-models"].split(","))
        )
        self.assertEqual(config["acceptance"]["expected_rows"], expected_rows)
        self.assertEqual(config["acceptance"]["promotion_mean_delta_min"], 0.005)
        self.assertEqual(config["acceptance"]["promotion_win_rate_min"], 0.6)
        self.assertEqual(config["acceptance"]["promotion_cluster_split_max_drop"], 0.01)


if __name__ == "__main__":
    unittest.main()
