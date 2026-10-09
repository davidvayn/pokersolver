import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from training_coverage import select_families, extra_calibration, extension_families
from run_native_value_preflight import atomic_json, sha256


class CoverageTests(unittest.TestCase):
    def root(self, board, leaves=5):
        return {"public": {"board": board, "public_history":
            ["Preflop:p0:raise_to_2.000bb", "Preflop:p1:raise_to_7.500bb"]}, "turn_leaf_count": leaves}

    def test_scores_blind_distinct_strata_and_exclusions(self):
        from native_value_dataset import board_family
        roots = [self.root([6, 49, 11]), self.root([30, 38, 51]), self.root([29, 6, 51])]
        self.assertEqual(select_families(roots, set()), [0, 1, 2])
        with self.assertRaises(ValueError):
            select_families(roots, {board_family(roots[0]["public"]["board"])})
        roots[0]["turn_leaf_count"] = 9
        with self.assertRaises(ValueError):
            select_families(roots, set())

    def test_extra_search_states_cannot_change_affine_actions(self):
        group = SimpleNamespace(history=[], actor=0, actions=["check", "bet"],
            terminal=np.ones((2, 2)), coefficients=np.ones((2, 1, 2)), target=np.ones((2, 2)),
            weights=np.ones(2), support=np.ones(2, dtype=bool))
        labels = [dict(board=[6, 49, 11, 0])]
        names = ["early", "middle", "late", "final_average"]
        capture = dict(source_policy_sha256="a"*64, turn_iterations=64, flop_iterations=128,
            capture_selection="stratified_learned_search_and_final_average_beliefs_with_native_labels",
            targets=[dict(board=[6, 49, 11, 1], state_distribution=
                f"learned_flop_{'search_'+names[i%4] if i%4<3 else names[i%4]}_belief_native_label") for i in range(16)])
        padded, ordered = extra_calibration([group], labels, capture, "a"*64)
        self.assertEqual(len(ordered), 17)
        arbitrary = np.full((17, 2), 1e4); arbitrary[0] = [2, 3]
        np.testing.assert_array_equal(padded[0].backup(arbitrary), [[3, 4], [3, 4]])
        self.assertEqual(group.coefficients.shape, (2, 1, 2))
        bad = copy.deepcopy(capture); bad["targets"][0]["board"] = [0, 5, 10, 15]
        with self.assertRaises(ValueError):
            extra_calibration([group], labels, bad, "a"*64)
        with self.assertRaises(ValueError):
            extra_calibration([group], labels, capture, "b"*64)

    def test_registry_cannot_expand_training_without_hashed_roots_and_frozen_split(self):
        from native_value_dataset import board_family
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory); families=[]; pinned={}
            for i, board in enumerate(([6,49,11], [30,38,51], [29,6,51])):
                path=work/f"root-{100+i}.json"; atomic_json(path,self.root(board))
                digest=sha256(path); pinned[str(path)]=digest
                families.append(dict(root=100+i,role="TRAIN",family=list(board_family(board)),
                                     rootPath=str(path),rootSha256=digest))
            existing={(0,4,8)}; forbidden={(12,16,20)}
            plan=dict(schema="new-training-coverage-protocol-v1",status="complete",releaseAccepted=False,
                      corpusSha256="a"*64,splitReferenceSha256="b"*64,families=families,
                      excludedFamilies=[list(f) for f in existing|forbidden],pinnedInputs=pinned)
            path=work/"protocol.json"
            def call(p):
                atomic_json(path,p)
                return extension_families(dict(path=str(path),sha256=sha256(path)),"a"*64,"b"*64,existing,forbidden)
            self.assertEqual(len(call(plan)),3)
            for mutate in (lambda p: p.update(corpusSha256="c"*64),
                           lambda p: p["excludedFamilies"].pop(),
                           lambda p: p["families"][0].update(role="HOLDOUT"),
                           lambda p: p["families"][0].update(root=101),
                           lambda p: p["families"][0].update(rootSha256="c"*64),
                           lambda p: p["families"][0].update(family=list(next(iter(forbidden))))):
                bad=copy.deepcopy(plan); mutate(bad)
                with self.assertRaises(ValueError): call(bad)
            call(plan); receipt=dict(path=str(path),sha256=sha256(path))
            atomic_json(path,{**plan,"status":"failed"})
            with self.assertRaisesRegex(ValueError,"changed"):
                extension_families(receipt,"a"*64,"b"*64,existing,forbidden)


if __name__ == "__main__": unittest.main()
