import copy
import unittest

from run_completed_control_diagnosis import chance_turns, completed_case


class CompletedControlDiagnosisTests(unittest.TestCase):
    def test_reads_the_native_nested_state_board(self):
        candidate = dict(schema="hu-native-counterfactual-turn-flop-pilot-v1",
                         state=dict(street="flop", board=[31, 46, 48]))
        turns = chance_turns(candidate)
        self.assertEqual(len(turns), 49)
        self.assertFalse(set(turns) & {31, 46, 48})
        for board in ([31, 31, 48], [31, 46, 52], [True, 46, 48], []):
            with self.assertRaises(ValueError):
                chance_turns(dict(candidate, state=dict(street="flop", board=board)))

    def test_only_completed_audited_cases_from_matching_students(self):
        row = dict(spot="control", seed=100101, modelSha256="model")
        response = dict(schema="postflop-student-value-pilot-v1", status="rejected",
                        studentManifestSha256="students", cases=[row])
        students = dict(status="complete", predictions=[dict(
            seed=10601, modelSha256="model", maximumParityErrorBb=1e-5)])
        self.assertEqual(completed_case(response, students, "students", "control", 100101)[0], row)
        for change in (dict(status="running"), dict(cases=[]), dict(cases=[row, row]),
                       dict(studentManifestSha256="other")):
            with self.assertRaises(ValueError):
                completed_case(dict(response, **change), students, "students", "control", 100101)
        for value in (float("nan"), .01, -1):
            wrong = copy.deepcopy(students)
            wrong["predictions"][0]["maximumParityErrorBb"] = value
            with self.assertRaises(ValueError):
                completed_case(response, wrong, "students", "control", 100101)
        wrong = copy.deepcopy(students); wrong["predictions"][0]["seed"] = 10602
        with self.assertRaises(ValueError):
            completed_case(response, wrong, "students", "control", 100101)


if __name__ == "__main__": unittest.main()
