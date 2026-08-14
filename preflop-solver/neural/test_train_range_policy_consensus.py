import gzip
import json
import tempfile
import unittest
from pathlib import Path

import train_range_policy_consensus as consensus


def record(index: int, street: str, weight: float) -> dict:
    return {
        "record_type": "range_conditioned_average_strategy",
        "weight": weight,
        "state": {
            "street": street,
            "board": [index % 52, (index + 1) % 52, (index + 2) % 52],
            "public_history": [f"state-{index}"],
            "actor": index % 2,
        },
        "action_labels": ["check", "bet_to_1"],
    }


def write_dataset(
    path: Path, records: list[dict], subset_of_sha256: str | None = None
) -> None:
    metadata = {"record_type": "metadata", "records": len(records)}
    if subset_of_sha256 is not None:
        metadata["subset_of_sha256"] = subset_of_sha256
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write((json.dumps(metadata) + "\n").encode())
            for item in records:
                stream.write((json.dumps(item) + "\n").encode())


class ConsensusTrainingCapTests(unittest.TestCase):
    def test_release_holdout_states_are_excluded_from_training_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_records = [
                record(index, ("flop", "turn", "river")[index % 3], 100 - index)
                for index in range(30)
            ]
            heldout_records = [source_records[0], source_records[4], source_records[8]]
            source = root / "source.jsonl.gz"
            heldout = root / "heldout.jsonl.gz"
            output = root / "cap.jsonl.gz"
            write_dataset(source, source_records)
            write_dataset(heldout, heldout_records)

            consensus.training_cap(source, heldout, output, 12)

            excluded = {consensus.state_identity(item) for item in heldout_records}
            with gzip.open(output, "rt", encoding="utf-8") as stream:
                metadata = json.loads(next(stream))
                retained = [json.loads(line) for line in stream]
            self.assertEqual(metadata["records"], 12)
            self.assertEqual(
                metadata["excluded_release_holdout_sha256"], consensus.sha256(heldout)
            )
            self.assertTrue(
                all(consensus.state_identity(item) not in excluded for item in retained)
            )
            self.assertEqual(
                {item["state"]["street"] for item in retained},
                {"flop", "turn", "river"},
            )

    def test_validation_cap_preserves_release_corpus_ancestry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                record(index, ("flop", "turn", "river")[index % 3], 100 - index)
                for index in range(30)
            ]
            source = root / "heldout.jsonl.gz"
            output = root / "pilot.jsonl.gz"
            ancestor = "a" * 64
            write_dataset(source, records, ancestor)

            consensus.validation_cap(source, output, 12)

            with gzip.open(output, "rt", encoding="utf-8") as stream:
                metadata = json.loads(next(stream))
                retained = [json.loads(line) for line in stream]
            self.assertEqual(metadata["records"], 12)
            self.assertEqual(metadata["subset_of_sha256"], ancestor)
            self.assertEqual(
                metadata["validation_subset_of_sha256"], consensus.sha256(source)
            )
            self.assertEqual(
                {item["state"]["street"] for item in retained},
                {"flop", "turn", "river"},
            )


if __name__ == "__main__":
    unittest.main()
