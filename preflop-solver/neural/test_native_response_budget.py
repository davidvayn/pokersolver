import unittest

from run_native_value_response import validate_search_budget, select_control_profiles
from run_native_action_value_probe import select_student


class SearchBudgetTest(unittest.TestCase):
    def test_alternative_probe_preserves_seed_pairing_and_default_identity_checks(self):
        students=[dict(seed=10601+i,modelSha256=h*64,maximumParityErrorBb=1e-6)
                  for i,h in enumerate(("c","d"))]
        profile=dict(seed=100101,modelSha256="a"*64)
        with self.assertRaises(ValueError): select_student(profile,students)
        self.assertEqual(select_student(profile,list(reversed(students)),True),students[0])
        self.assertEqual(select_student({**profile,"modelSha256":"c"*64},students),students[0])
        for invalid in ([],students+students,[{**students[0],"maximumParityErrorBb":.1},students[1]]):
            with self.assertRaises(ValueError): select_student(profile,invalid,True)

    def test_updated_values_compare_at_the_same_128_budget_without_redundant_32_pilot(self):
        students={"predictions":[{"modelSha256":"c"*64},{"modelSha256":"d"*64}]}
        prior={"status":"complete","flopIterations":128,"trainingTurnIterations":64,"playedTurnIterations":64,
               "profiles":[{"seed":100101+i,"modelSha256":h*64,"comparison":{"learnedGainBb":.3}}
                           for i,h in enumerate(("a","b"))]}
        self.assertEqual(set(validate_search_budget(128,prior,students,updated_values=True)),{100101,100102})
        for invalid in (None,{**prior,"status":"running"},{**prior,"flopIterations":32}):
            with self.assertRaises(ValueError): validate_search_budget(128,invalid,students,updated_values=True)
        with self.assertRaises(ValueError): validate_search_budget(32,prior,students,updated_values=True)
        with self.assertRaises(ValueError): validate_search_budget(128,prior,students,prior,updated_values=True)

    def test_extra_updates_need_same_weights_and_completed_32_update_reference(self):
        students={"predictions":[{"modelSha256":"a"*64},{"modelSha256":"b"*64}]}
        prior={"status":"complete","flopIterations":32,"trainingTurnIterations":64,"playedTurnIterations":64,
               "profiles":[{"seed":100101+i,"modelSha256":p["modelSha256"],"comparison":{"learnedGainBb":.4}}
                           for i,p in enumerate(students["predictions"])]}
        self.assertEqual(validate_search_budget(32,None,students),{})
        self.assertEqual(set(validate_search_budget(128,prior,students)),{100101,100102})
        with self.assertRaises(ValueError): validate_search_budget(128,None,students)
        with self.assertRaises(ValueError): validate_search_budget(128,{**prior,"status":"running"},students)
        with self.assertRaises(ValueError): validate_search_budget(128,{**prior,"flopIterations":128},students)
        with self.assertRaises(ValueError): validate_search_budget(128,prior,{"predictions":[{"modelSha256":"c"*64},students["predictions"][1]]})

    def test_transfer_pins_successful_128_pair_without_requiring_redundant_new_32_run(self):
        students={"predictions":[{"modelSha256":"a"*64},{"modelSha256":"b"*64}]}
        transfer={"status":"complete","flopIterations":128,"trainingTurnIterations":64,"playedTurnIterations":64,
                  "bothSeedsRetainOrImproveResponse":True,
                  "profiles":[{"seed":100101+i,"modelSha256":p["modelSha256"],"comparison":{"learnedGainBb":.2}}
                              for i,p in enumerate(students["predictions"])]}
        self.assertEqual(validate_search_budget(128,None,students,transfer),{})
        with self.assertRaises(ValueError): validate_search_budget(32,None,students,transfer)
        with self.assertRaises(ValueError): validate_search_budget(128,None,students,{**transfer,"bothSeedsRetainOrImproveResponse":False})
        with self.assertRaises(ValueError): validate_search_budget(128,transfer,students,transfer)
        with self.assertRaises(ValueError): validate_search_budget(128,None,{"predictions":[{"modelSha256":"c"*64},students["predictions"][1]]},transfer)

    def test_control_selection_is_explicit_and_excludes_old_iteration_arms(self):
        profiles=[dict(context=c,seed=s,iterations=i) for c in ("start","facing") for s in (100101,100102) for i in (8,32)]
        control={"status":"complete","profiles":profiles}
        self.assertEqual(select_control_profiles(control,"facing"),[p for p in profiles if p["context"]=="facing" and p["iterations"]==32])
        with self.assertRaises(ValueError): select_control_profiles(control,None)
        with self.assertRaises(ValueError): select_control_profiles(control,"missing")
        with self.assertRaises(ValueError): select_control_profiles({**control,"status":"running"},"facing")
