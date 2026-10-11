import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from capture_cash_flop_leaf_dataset import capture
from cash_turn_roots import root_fingerprint
from cash_value_dataset import build_dataset, validate_dataset
from native_value_dataset import compatible_masses, legal_combos
from run_cash_flop_pilot import sha
import test_cash_flop_leaves as leaf_tests
from test_cash_turn_roots import fixture
from test_cash_value_dataset import complete_fixture_targets, label_fixture


class CashFlopCoverageCaptureTests(unittest.TestCase):
    def test_option_is_strict_before_artifact_io(self):
        for invalid in (None, 0, 1, "both", [], {}):
            with self.assertRaisesRegex(ValueError, "explicit boolean"):
                capture(*[Path("missing") for _ in range(7)], include_after_check=invalid)

    def test_real_capture_path_preserves_legacy_inputs_and_pins_three_action_lineage(self):
        root, solution = leaf_tests.CashFlopLeafTests().both_seat_bets()
        source, model = fixture()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            def save(name, value):
                path = base / name
                path.write_text(json.dumps(value))
                return path
            binary = base / "binary"; binary.write_bytes(b"native")
            policy = save("policy.json", model)
            source.update(schema="hu-cash-authentic-flop-roots-v1", policy_sha256=sha(policy), roots=[root])
            ranges = np.asarray(root["solve_input"]["state"]["ranges"])
            root["compatible_joint_mass"] = float((ranges * compatible_masses(ranges)).sum(axis=1)[0])
            root["root_sha256"] = root_fingerprint(sha(policy), root)
            roots = save("roots.json", source)
            excluded = copy.deepcopy(source)
            excluded_root = excluded["roots"][0]
            excluded_state = excluded_root["solve_input"]["state"]
            excluded_state["board"] = [40, 45, 2]
            legal = legal_combos(excluded_state["board"])
            excluded_state["ranges"] = np.tile(legal / legal.sum(), (2, 1)).tolist()
            excluded_root["root_sha256"] = root_fingerprint(sha(policy), excluded_root)
            excluded_path = save("excluded.json", excluded)
            solution.update(rules_sha256=source["rules_sha256"], value_network_sha256="a" * 64)
            frozen = save("frozen.json", solution)
            saved = dict(path=str(frozen), root_sha256=root["root_sha256"], value_network_sha256="a" * 64,
                         native_binary_sha256=sha(binary), solution_sha256=sha(frozen))
            pair = save("pair.json", dict(roots_sha256=sha(roots), source_policy_sha256=sha(policy),
                        native_binary_sha256=sha(binary), status="research_only", active=False,
                        value_network_sha256s=["a" * 64, "b" * 64], captures=[saved, saved]))
            original_paths = []
            for i, board in enumerate(([48, 45, 26, 3], [32, 29, 18, 7], [44, 41, 6, 11])):
                label = label_fixture(); legal = legal_combos(board)
                label["input"]["game"] = copy.deepcopy(root["solve_input"]["game"])
                ranges = np.tile(legal / legal.sum(), (2, 1))
                label["input"]["state"].update(board=board, ranges=ranges.tolist())
                label["opponent_compatible_mass"] = compatible_masses(ranges).tolist()
                label["counterfactual_values_bb"] = np.tile(legal * -.86, (2, 1)).tolist()
                complete_fixture_targets(label)
                original_paths.append(save(f"original-{i}.json", label))
            split = save("split.json", build_dataset(original_paths))

            def native(command, **kwargs):
                config = json.loads(Path(command[command.index("--input") + 1]).read_text())
                label = label_fixture(); legal = legal_combos(config["state"]["board"])
                label.update(input=config, joint_iterations=config["iterations"],
                             counterfactual_values_bb=np.tile(legal * -.04, (2, 1)).tolist(),
                             opponent_compatible_mass=compatible_masses(np.asarray(config["state"]["ranges"])).tolist())
                label["metrics"]["cash"]["expected_house_rake_bb"] = .08
                complete_fixture_targets(label)
                Path(command[command.index("--output") + 1]).write_text(json.dumps(label))

            with patch("probe_cash_flop_leaves.subprocess.run", side_effect=native):
                legacy = capture(binary, roots, policy, pair, base / "legacy", split, excluded_path)
                explicit = capture(binary, roots, policy, pair, base / "explicit", split, excluded_path,
                                   include_after_check=False)
                both = capture(binary, roots, policy, pair, base / "both", split, excluded_path,
                               include_after_check=True)
            self.assertEqual(legacy["contexts"], 4)
            self.assertEqual(both["contexts"], 6)
            # Identical byte output when the extension is disabled.
            self.assertEqual(legacy, explicit)
            old = json.loads((base / "legacy/dataset.json").read_text())
            added = json.loads((base / "both/dataset.json").read_text())
            validate_dataset(added)
            self.assertEqual(old["labels"], added["labels"][:4])
            self.assertEqual(old["flop_leaf_provenance"]["schema"], "hu-cash-frozen-flop-leaf-lineage-v1")
            self.assertNotIn("branch_coverage", old["flop_leaf_provenance"])
            self.assertEqual(added["flop_leaf_provenance"]["schema"], "hu-cash-frozen-flop-leaf-lineage-v2")
            self.assertEqual(both["branch_coverage"], "both-seats-small-bet-v1")
            for completed in (base / "legacy", base / "both"):
                with self.assertRaisesRegex(ValueError, "immutable output"):
                    capture(binary, roots, policy, pair, completed, split, excluded_path,
                            include_after_check=True)
            for row in added["flop_leaf_provenance"]["rows"][-2:]:
                self.assertEqual(row["branch_labels"], ["check", "bet_to_2.480bb", "call"])
                self.assertEqual([r["actor"] for r in row["branch_policy_rows"]], [1, 0, 1])
            # A diagnosed family cannot sneak back into coverage through the
            # new branch. Omit it before any labeling, not after training.
            forbidden = save("excluded-actual.json", source)
            with patch("probe_cash_flop_leaves.subprocess.run") as native_call:
                with self.assertRaisesRegex(ValueError, "no unheldout board families"):
                    capture(binary, roots, policy, pair, base / "forbidden", split, forbidden,
                            include_after_check=True)
                native_call.assert_not_called()


if __name__ == "__main__": unittest.main()
