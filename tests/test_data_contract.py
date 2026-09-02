from __future__ import annotations

import csv
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/proxy2000_v2"
FULL = DATA / "fad_proxy2000_v2_full.csv"
NUPACK_FEATURES = DATA / "nupack_features_full.csv"
EXPECTED_SHA256 = "0876C2B25A5571B834EBAC2C0C0F7671E023F6028688E90CCBA884A75FD4322A"
EXPECTED_NUPACK_SHA256 = "EFF085664F4377CCBA8F8D0E1CD4842BD8C24DC5FA5E56F476B28E736BD8A32D"
REQUIRED_COLUMNS = {"Full", "MFE", "Pre", "Aft", "Sequence", "Gap1", "Gap2", "Gap3", "dock", "gbsa", "label"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


class DataContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full_rows = read_rows(FULL)

    def test_canonical_file_hash_and_row_count(self):
        self.assertEqual(file_sha256(FULL), EXPECTED_SHA256)
        self.assertEqual(len(self.full_rows), 2000)

    def test_schema_and_unique_sequences(self):
        self.assertEqual(set(self.full_rows[0]), REQUIRED_COLUMNS)
        sequences = [row["Sequence"] for row in self.full_rows]
        self.assertEqual(len(sequences), len(set(sequences)))

    def test_label_formula(self):
        max_error = 0.0
        for row in self.full_rows:
            expected = 0.961 * float(row["Gap2"]) + 0.039 * float(row["gbsa"])
            max_error = max(max_error, abs(float(row["label"]) - expected))
        self.assertLessEqual(max_error, 0.01)

    def test_historical_split_partition(self):
        splits = {
            "train": read_rows(DATA / "train.csv"),
            "val": read_rows(DATA / "val.csv"),
            "test": read_rows(DATA / "test.csv"),
        }
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 1600, "val": 200, "test": 200})
        sequence_sets = {name: {row["Sequence"] for row in rows} for name, rows in splits.items()}
        self.assertFalse(sequence_sets["train"] & sequence_sets["val"])
        self.assertFalse(sequence_sets["train"] & sequence_sets["test"])
        self.assertFalse(sequence_sets["val"] & sequence_sets["test"])
        self.assertEqual(set.union(*sequence_sets.values()), {row["Sequence"] for row in self.full_rows})

    def test_nupack_supplement_has_safe_schema_and_complete_coverage(self):
        nupack_rows = read_rows(NUPACK_FEATURES)
        self.assertEqual(file_sha256(NUPACK_FEATURES), EXPECTED_NUPACK_SHA256)
        self.assertEqual(len(nupack_rows), 2000)
        self.assertEqual(len(nupack_rows[0]), 1193)
        self.assertEqual(
            {row["Sequence"] for row in nupack_rows},
            {row["Sequence"] for row in self.full_rows},
        )
        feature_columns = set(nupack_rows[0]) - {"Sequence"}
        self.assertTrue(feature_columns)
        self.assertTrue(
            all(name.startswith(("nupack_", "delta_nupack_")) for name in feature_columns)
        )
        self.assertFalse(REQUIRED_COLUMNS - {"Sequence"} & set(nupack_rows[0]))


if __name__ == "__main__":
    unittest.main()
