import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from run_on_policy_calibration_probe import measure, shifted_enough, validate_capture


class OnPolicyCalibrationProbeTests(unittest.TestCase):
    def test_feature_analysis_runs_in_a_guarded_child_not_the_controller(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder)
            binary,capture=work/"binary",work/"capture"
            binary.write_bytes(b"binary"); capture.write_bytes(b"capture")
            stop=threading.Event()
            def completed(command,env,output,seconds,memory,event):
                self.assertEqual(command[-2],"--measure-request")
                self.assertEqual(seconds,600)
                self.assertEqual(memory,6*1024**3)
                self.assertIs(event,stop)
                request=json.loads(Path(command[-1]).read_text())
                self.assertEqual(request["capture"],str(capture))
                Path(request["result"]).write_text(json.dumps(dict(authenticRmseBb=.3)))
                return dict(status="complete",sampledPeakMemoryBytes=123)
            with patch("run_on_policy_calibration_probe.guarded",side_effect=completed), \
                    patch("run_on_policy_calibration_probe.measure_values",side_effect=AssertionError("unbounded analysis")):
                result=measure(binary,capture,dict(path="model",sha256="sha"),work,work/"cache",stop)
            self.assertEqual(result["authenticRmseBb"],.3)
            self.assertEqual(result["analysisWorker"]["sampledPeakMemoryBytes"],123)

    def test_screen_needs_three_complete_comparisons_and_two_shifted_families(self):
        results=[dict(root=root,old=dict(authenticRmseBb=1.),current=dict(authenticRmseBb=value))
            for root,value in zip((100,101,102),(1.2,1.11,.9))]
        self.assertTrue(shifted_enough(results))
        results[1]["current"]["authenticRmseBb"]=1.01
        self.assertFalse(shifted_enough(results))
        with self.assertRaises(ValueError): shifted_enough(results[:2])
        results[0]["current"]["authenticRmseBb"]=float("nan")
        with self.assertRaises(ValueError): shifted_enough(results)

    def test_capture_cannot_reuse_another_root_model_seed_or_native_budget(self):
        family=dict(rootSha256="root",solverSeed=100101,sampleSeed=12,family=[0,5,10])
        source=dict(source_public_input_sha256="root",source_policy_sha256="policy",
            proposal_model_sha256="model",seed=100101,sampling_seed=12,
            proposal_turn_iterations=64,native_label_queries=16,policy_observation_parity_checked=True,
            turn_iterations=64,flop_iterations=128,
            capture_selection="stratified_learned_search_and_final_average_beliefs_with_native_labels",
            targets=[dict(board=[0,5,10,15],state_distribution=band) for band in (
                "learned_flop_search_early_belief_native_label","learned_flop_search_middle_belief_native_label",
                "learned_flop_search_late_belief_native_label","learned_flop_final_average_belief_native_label")]*4)
        validate_capture(source,family,"model","policy")
        for key,value in (("seed",100102),("source_public_input_sha256","other"),
                          ("proposal_model_sha256","other"),("native_label_queries",15),
                          ("turn_iterations",256),("policy_observation_parity_checked",False)):
            wrong=copy.deepcopy(source); wrong[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_capture(wrong,family,"model","policy")


if __name__=="__main__": unittest.main()
