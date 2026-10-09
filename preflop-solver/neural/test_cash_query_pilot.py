import copy
import unittest
from run_cash_query_pilot import validate_result


class CashQueryPilotTests(unittest.TestCase):
    def fixture(self):
        request = {"rulesSha256":"r"*64,"networkSha256":"n"*64,"query":{
            "requestId":"one","modelVersion":"cash-research","depthBb":20,"stateHash":"s"*64}}
        result = {"schema":"hu-cash-average-policy-result-v1","validationStatus":"research_only",
            "rulesSha256":request["rulesSha256"],"networkSha256":request["networkSha256"],**request["query"],
            "rolloutSamplesPerAction":128,"maximumAccountingResidualBb":0.,"maximumProbabilitySumError":0.,
            "evEvaluation":"paired-full-hand-monte-carlo","exactPolicyTreeNodes":0,
            "exactAllInChanceOutcomes":0,"actionEvEvaluations":["paired-full-hand-monte-carlo"],
            "actions":[{"probability":1.,"evBb":-.4,"standardErrorBb":0.,"confidence":"low"}]}
        return request,result

    def test_reference_rows_keep_research_status_and_own_negative_values(self):
        request,result = self.fixture(); validate_result(result,request,128)
        for key,value in [("validationStatus","accepted"),("networkSha256","other"),("rulesSha256","other"),
                          ("stateHash","other"),("rolloutSamplesPerAction",32),("maximumAccountingResidualBb",.1)]:
            with self.assertRaises(ValueError): validate_result(dict(result,**{key:value}),request,128)

    def test_invalid_probability_ev_or_uncertainty_cannot_be_cached_as_valid(self):
        request,result = self.fixture()
        for key,value in [("evBb",float("nan")),("evBb",21),("standardErrorBb",-.1),
                          ("probability",1.1),("probability",.5),("confidence","high")]:
            bad = copy.deepcopy(result); bad["actions"][0][key] = value
            with self.assertRaises(ValueError): validate_result(bad,request,128)

    def test_exact_river_has_no_sampled_fallback_or_claim_of_equilibrium(self):
        request,result = self.fixture()
        request["query"]["street"] = "river"
        result.update(evEvaluation="exact-river-policy-expectation",exactPolicyTreeNodes=1980,rolloutSamplesPerAction=0)
        result["actionEvEvaluations"] = ["exact-river-policy-expectation"]
        validate_result(result,request,128)
        bad = copy.deepcopy(result); bad["actions"][0]["standardErrorBb"] = .01
        with self.assertRaises(ValueError): validate_result(bad,request,128)
        request["query"]["street"] = "turn"
        with self.assertRaises(ValueError): validate_result(result,request,128)

    def test_mixed_exact_all_ins_remain_bounded_and_explicit(self):
        request,result = self.fixture(); request["query"]["street"] = "turn"
        result.update(evEvaluation="paired-monte-carlo-with-exact-all-ins",exactAllInChanceOutcomes=45540,
            actionEvEvaluations=["exact-closed-all-in-expectation"])
        validate_result(result,request,128)
        for key,value in [("exactAllInChanceOutcomes",2_000_001),("actionEvEvaluations",[]),
            ("evEvaluation","paired-full-hand-monte-carlo")]:
            with self.assertRaises(ValueError): validate_result(dict(result,**{key:value}),request,128)
        bad = copy.deepcopy(result); bad["actions"][0]["standardErrorBb"] = .1
        with self.assertRaises(ValueError): validate_result(bad,request,128)


if __name__ == "__main__": unittest.main()
