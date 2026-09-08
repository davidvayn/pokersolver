import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from cloud_blueprint_run import CANONICAL_HAND_CLASSES, combo_weight
from run_compact_preflop_continuation import compare, main
from run_native_value_preflight import sha256


class CompactComparisonTests(unittest.TestCase):
    def fixture(self, seed):
        rows = []
        for group in range(100):
            for hand in sorted(CANONICAL_HAND_CLASSES):
                rows.append(dict(key=f"{group}-{hand}", actor=group%2,
                    history=["blinds"] + ([str(group)] if group else []),
                    hand=hand, comboWeight=combo_weight(hand), actions=["fold","call"],
                    probabilities=[0.2,0.8], averageVisits=2, regretUpdates=6, trained=True))
        return dict(schema="compact-preflop-continuation-pilot-v1", releaseAccepted=False,
                    totalNodes=16900, rows=rows, config=dict(seed=seed), valueModelSha256="a"*64)

    def test_preserves_established_gate_and_requires_complete_trained_rows(self):
        a,b = self.fixture(27001),self.fixture(27002)
        result = compare([a,b])
        self.assertTrue(result["establishedRootStability"]["passed"])
        self.assertEqual(result["withinPublicState"]["completePublicStates"],100)
        for mutation in ("missing", "untrained", "model", "seed", "baseline", "sampling", "proposal", "history_baseline", "turn_baseline", "scale", "simultaneous", "turn_averages", "played_targets"):
            bad = copy.deepcopy(b)
            if mutation=="missing": bad["rows"].pop()
            elif mutation=="untrained": bad["rows"][0]["trained"]=False
            elif mutation=="model": bad["valueModelSha256"]="b"*64
            elif mutation=="baseline": bad["exactCheckdownSha256"]="c"*64
            elif mutation=="sampling": bad["endpointSampling"]="root_stratified"
            elif mutation=="proposal": bad["endpointProposalSha256"]="d"*64
            elif mutation=="history_baseline": bad["historyBaseline"]=True
            elif mutation=="turn_baseline": bad["completeTurnBaseline"]=True
            elif mutation=="scale": bad["flopCheckdownScale"]=2.0
            elif mutation=="simultaneous": bad["simultaneousUpdates"]=True
            elif mutation=="turn_averages": bad["rootRealizationTurnAverages"]=True
            elif mutation=="played_targets": bad["playedProfileTargets"]=True
            else: bad["config"]["seed"]=27001
            with self.assertRaises(ValueError): compare([a,bad])

    def test_parallel_controller_keeps_seed_identity_and_order(self):
        barrier = threading.Barrier(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, model = root/"binary", root/"model"
            binary.write_bytes(b"fixture binary")
            model.write_bytes(b"fixture model")
            def worker(command, environment, output, *args):
                barrier.wait(timeout=5)  # Fails if the two jobs run serially.
                seed = int(environment["POKER_COMPACT_SEED"])
                frozen = root/f"seed{seed}.gz"
                frozen.write_bytes(str(seed).encode())
                result = dict(config=dict(seed=seed,iterations=int(environment["POKER_COMPACT_ROUNDS"])),continuationSeed=28001,
                    valueModelSha256=sha256(model),frozenPolicy=str(frozen),
                    frozenPolicySha256=sha256(frozen),progress=[dict(round=i+1) for i in range(int(environment["POKER_COMPACT_ROUNDS"]))],trainingSeconds=0.1,totalNodes=16900,
                    exactCheckdownSha256=environment.get("POKER_COMPACT_CHECKDOWN_SHA"),
                    endpointSampling=environment["POKER_COMPACT_ENDPOINT_SAMPLING"],
                    endpointProposalSha256=environment.get("POKER_COMPACT_PROPOSAL_SHA"),
                    checkpointInterval=int(environment["POKER_COMPACT_CHECKPOINT_INTERVAL"]),
                    resumedFromRound=json.loads(Path(environment["POKER_COMPACT_RESUME_RECEIPT"]).read_text())["completedIterations"] if "POKER_COMPACT_RESUME_RECEIPT" in environment else 0,
                    resumeReceiptSha256=environment.get("POKER_COMPACT_RESUME_SHA"),
                    completeTurnBaseline=environment["POKER_COMPACT_TURN_BASELINE"]=="1",
                    playedProfileTargets=environment["POKER_COMPACT_PLAYED_TARGETS"]=="1",
                    rootUpdateTraceEnabled=environment["POKER_COMPACT_TRACE_ROOT"]=="1")
                Path(environment["POKER_COMPACT_OUTPUT"]).write_text(json.dumps(result))
                return dict(status="complete")
            argv = ["pilot", "--binary", str(binary), "--binary-sha256", sha256(binary),
                "--model", str(model), "--model-sha256", sha256(model),
                "--rounds", "2", "--workers", "2", "--output", str(root/"run")]
            with patch("sys.argv",argv), patch("run_compact_preflop_continuation.guarded",side_effect=worker), \
                    patch("run_compact_preflop_continuation.signal.signal"), \
                    patch("run_compact_preflop_continuation.compare",return_value={}) as comparison:
                main()
            manifest = json.loads((root/"run/manifest.json").read_text())
            self.assertEqual(manifest["status"],"complete")
            self.assertEqual(manifest["maximumConcurrentWorkers"],2)
            self.assertEqual([j["seed"] for j in manifest["jobs"]],[27001,27002])
            self.assertEqual([r["config"]["seed"] for r in comparison.call_args.args[0]],[27001,27002])
            kernel = root/"kernel"
            kernel.write_bytes(b"fixture kernel")
            traced = argv[:-1]+[str(root/"traced"), "--checkdown", str(kernel),
                "--checkdown-sha256", sha256(kernel), "--turn-baseline", "--played-profile-targets",
                "--trace-root-updates"]
            with patch("sys.argv",traced), patch("run_compact_preflop_continuation.guarded",side_effect=worker), \
                    patch("run_compact_preflop_continuation.signal.signal"), \
                    patch("run_compact_preflop_continuation.compare",return_value={}) as comparison:
                main()
            self.assertTrue(json.loads((root/"traced/manifest.json").read_text())["rootUpdateTraceEnabled"])
            self.assertTrue(all(r["rootUpdateTraceEnabled"] for r in comparison.call_args.args[0]))
            proposal = root/"proposal.json"
            proposal.write_bytes(b"fixture proposal")
            importance = [str(root/"importance") if x == str(root/"traced") else x for x in traced]
            importance += ["--endpoint-sampling", "fixed_importance", "--endpoint-proposal", str(proposal),
                           "--endpoint-proposal-sha256", sha256(proposal)]
            with patch("sys.argv",importance), patch("run_compact_preflop_continuation.guarded",side_effect=worker), \
                    patch("run_compact_preflop_continuation.signal.signal"), \
                    patch("run_compact_preflop_continuation.compare",return_value={}):
                main()
            manifest = json.loads((root/"importance/manifest.json").read_text())
            self.assertEqual(manifest["endpointProposalSha256"], sha256(proposal))
            self.assertEqual(manifest["maximumEndpointsPerRound"], 1)
            extended = [str(root/"extended") if x == str(root/"importance") else x
                        for x in importance if x != "--trace-root-updates"]
            extended[extended.index("--rounds")+1] = "128"
            with patch("sys.argv",extended), patch("run_compact_preflop_continuation.guarded",side_effect=worker), \
                    patch("run_compact_preflop_continuation.signal.signal"), \
                    patch("run_compact_preflop_continuation.compare",return_value={}):
                main()
            manifest = json.loads((root/"extended/manifest.json").read_text())
            self.assertEqual(manifest["maximumWorkerSeconds"], 3600)
            self.assertEqual(manifest["rounds"], 128)
            self.assertFalse(manifest["rootUpdateTraceEnabled"])
            checkpoint = root/"round0008.mpk.gz"
            checkpoint.write_bytes(b"fixture checkpoint")
            receipt = root/"round0008.json"
            receipt.write_text(json.dumps(dict(schema="compact-preflop-checkpoint-v1",
                completedIterations=8,checkpoint=checkpoint.name,checkpointSha256=sha256(checkpoint))))
            recovered = [str(root/"recovered") if x==str(root/"extended") else x for x in extended]
            recovered += ["--checkpoint-interval","8","--maximum-worker-seconds","5400",
                          "--resume","27001",str(receipt),sha256(receipt)]
            with patch("sys.argv",recovered), patch("run_compact_preflop_continuation.guarded",side_effect=worker), \
                    patch("run_compact_preflop_continuation.signal.signal"), \
                    patch("run_compact_preflop_continuation.compare",return_value={}):
                main()
            manifest = json.loads((root/"recovered/manifest.json").read_text())
            self.assertEqual(manifest["maximumWorkerSeconds"],5400)
            self.assertEqual(manifest["resumedSeeds"]["27001"]["round"],8)
            barrier = threading.Barrier(2)
            def failing_worker(command, environment, output, *args):
                barrier.wait(timeout=5)
                if environment["POKER_COMPACT_SEED"] == "27001":
                    raise ValueError("injected worker failure")
                self.assertTrue(args[-1].wait(timeout=5),"sibling must receive stop")
                raise ValueError("sibling stopped")
            argv[-1] = str(root/"failed")
            with patch("sys.argv",argv), patch("run_compact_preflop_continuation.guarded",side_effect=failing_worker), \
                    patch("run_compact_preflop_continuation.signal.signal"), \
                    patch("run_compact_preflop_continuation.compare") as comparison:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code,1)
                comparison.assert_not_called()
            manifest = json.loads((root/"failed/manifest.json").read_text())
            self.assertEqual(manifest["status"],"failed")
            self.assertFalse(manifest["releaseAccepted"])


if __name__=="__main__": unittest.main()
