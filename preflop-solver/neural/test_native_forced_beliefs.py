import copy
import unittest

from run_native_forced_beliefs import diagnostic_metrics
from test_native_value_dataset import corpus


class ForcedBeliefTests(unittest.TestCase):
    def test_explicit_intervention_provenance_and_real_value_errors(self):
        source=corpus()
        source["targets"]=[copy.deepcopy(source["targets"][0]) for _ in range(4)]
        for target in source["targets"]:
            target["state_distribution"]="explicit_forced_root_action_public_belief_native_label"
        source.update(maximum_states=4,observed_queries=4,source_policy_unchanged=True,
                      releaseAccepted=False,capture_selection="explicit_forced_root_actions_diagnostic_only",
                      predictions=[copy.deepcopy(t["counterfactual_values_bb"]) for t in source["targets"]],
                      interventions=[dict(forced_action_probability=1,source_action_frequency=.1,
                                          intervened_proposer_sha256="a"*64,public_belief_sha256="b"*64) for _ in range(4)])
        self.assertEqual(diagnostic_metrics(source)["weightedRmseBb"],0)
        for key,value in (("source_policy_unchanged",False),("releaseAccepted",True),("predictions",[])):
            changed=copy.deepcopy(source)
            changed[key]=value
            with self.assertRaises(ValueError): diagnostic_metrics(changed)
        changed=copy.deepcopy(source)
        changed["interventions"][0]["forced_action_probability"]=.5
        with self.assertRaises(ValueError): diagnostic_metrics(changed)
        changed=copy.deepcopy(source)
        changed["interventions"][0]["public_belief_sha256"]="missing"
        with self.assertRaises(ValueError): diagnostic_metrics(changed)


if __name__=="__main__": unittest.main()
