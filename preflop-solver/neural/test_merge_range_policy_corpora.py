import gzip
import json
import tempfile
import unittest
from pathlib import Path

import merge_range_policy_corpora as module


def write_corpus(path: Path, offset: int, records: int = 1) -> None:
    metadata = {
        "record_type": "metadata",
        "schema": module.DATASET_SCHEMA,
        "seed": 7,
        "records": records,
        "depth_bb": 20.0,
        "feature_schema": "features",
        "teacher": {
            "schema": "teacher",
            "rootOffset": offset,
            "roots": 1,
            "turnLeaves": 4,
            "sourcePolicySha256": "a" * 64,
            "flopConvergence": [{"root": offset, "value": offset}],
            "flopRangeResponse": [{"root": offset, "value": offset}],
            "validation": {"status": "accepted_for_training"},
        },
    }
    with gzip.open(path, "wt") as destination:
        destination.write(json.dumps(metadata) + "\n")
        for index in range(records):
            destination.write(
                json.dumps(
                    {
                        "record_type": "range_conditioned_average_strategy",
                        "state": {"board": [offset, index]},
                        "action_labels": ["check", "bet"],
                    }
                )
                + "\n"
            )


class MergeRangePolicyCorporaTests(unittest.TestCase):
    def test_merges_contiguous_validated_windows_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, output = root / "a.gz", root / "b.gz", root / "out.gz"
            write_corpus(first, 2, 2)
            write_corpus(second, 3, 1)
            report = module.merge([first, second], output)
            self.assertEqual(report["records"], 3)
            self.assertEqual(report["rootOffset"], 2)
            self.assertEqual(report["roots"], 2)
            with gzip.open(output, "rt") as source:
                metadata = json.loads(next(source))
                rows = list(source)
            self.assertEqual(metadata["records"], 3)
            self.assertEqual(metadata["teacher"]["roots"], 2)
            self.assertEqual(metadata["teacher"]["turnLeaves"], 8)
            self.assertEqual(len(rows), 3)

            second_output = root / "out-2.gz"
            second_report = module.merge([first, second], second_output)
            self.assertEqual(report["sha256"], second_report["sha256"])

    def test_rejects_noncontiguous_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a.gz", root / "b.gz"
            write_corpus(first, 2)
            write_corpus(second, 4)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                module.merge([first, second], root / "out.gz")


if __name__ == "__main__":
    unittest.main()
