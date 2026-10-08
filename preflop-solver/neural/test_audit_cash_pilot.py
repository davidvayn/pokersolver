import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from audit_cash_pilot import audit, load_run
from cash_profiles import CASH_NETWORK_SCHEMA, CASH_STATE_FEATURE_COUNT, CASH_STATE_FEATURE_SCHEMA, PAYOFF_CONTRACT, profile_rules, rules_digest


def fixtures():
    rules = profile_rules("nl25")
    runs, full, responses = [], [], []
    for seed in (7201, 7202):
        run = dict(config=dict(seed=seed, cash_profile="nl25", depth_bb=20), rules=rules,
                   action_abstraction={"grid": "test"}, rounds=32, training_hours=.01,
                   weights_bytes=100, weights_sha256=hashlib.sha256(str(seed).encode()).hexdigest())
        game = dict(cash_rules=rules, effective_stack_bb=20, small_blind_bb=.4, big_blind_bb=1,
                    action_abstraction=run["action_abstraction"])
        common = dict(validation_status="research_only", game=game, rules_sha256=rules_digest(rules),
                      policy_sha256=run["weights_sha256"], seed=929, deals=2, confidence=.99)
        ledger = dict(own_payoffs_bb=[.1, -.3], house_rake_bb=.2, maximum_accounting_residual_bb=0.)
        sample = []
        for seat in (0, 1):
            response = copy.deepcopy(ledger)
            response["own_payoffs_bb"][seat] += 1.
            response["own_payoffs_bb"][1-seat] -= 1.
            sample.append(dict(baseline=copy.deepcopy(ledger), response=response, own_gain_bb=1.))
        margin = 80 * math.sqrt(math.log(100)/4)
        responses.append(dict(common, schema="hu-cash-causal-sample-game-nash-conv-v1", samples=[sample, copy.deepcopy(sample)],
                              sample_mean_total_nash_conv_bb=2., one_sided_hoeffding_margin_bb=margin,
                              abstract_game_optimistic_upper_bound_bb=min(80., 2.+margin)))
        baseline = dict(mean_own_payoff_bb=ledger["own_payoffs_bb"], mean_house_rake_bb=.2, maximum_accounting_residual_bb=0.)
        full.append(dict(common, schema="hu-cash-full-hand-restricted-evaluation-v1", baseline=baseline,
                         comparison_baseline=copy.deepcopy(baseline), comparison_action_frequency_mae=.04,
                         comparison_primary_agreement=.86, comparison_aggregate_action_deltas={"fold":.01},
                         diagnostics=dict(comparison_rows=10, valid_probability_rows=10),
                         restricted_responses=[dict(mean_gain_bb=.01, maximum_accounting_residual_bb=0.)]))
        runs.append(run)
    for index in (0, 1):
        full[index]["comparison_sha256"] = runs[1-index]["weights_sha256"]
    return runs, full, responses


class CashPilotAuditTests(unittest.TestCase):
    def test_stability_and_inference_passes_never_activate_a_short_pilot(self):
        report = audit(*fixtures())
        self.assertFalse(report["release_eligible"])
        self.assertEqual(report["validation_status"], "research_only")
        gates = report["gates"]
        self.assertEqual(gates["cross_seed_frequency_mae"]["status"], "passed")
        self.assertEqual(gates["measured_training_duration"]["status"], "failed")
        self.assertEqual(gates["total_deviation_bound"]["status"], "failed")
        for name in ("policy_lookup_coverage", "quantized_probability_sums", "reach_weighted_action_ev_precision", "projected_total_hosted_storage", "full_routed_serving"):
            self.assertEqual(gates[name]["status"], "unmeasured")

    def test_worst_seed_distribution_controls_the_stability_gates(self):
        runs, full, responses = fixtures()
        full[1].update(comparison_action_frequency_mae=.08, comparison_primary_agreement=.70,
                       comparison_aggregate_action_deltas={"fold":.04})
        gates = audit(runs, full, responses)["gates"]
        for name in ("cross_seed_frequency_mae", "primary_action_agreement", "maximum_aggregate_action_delta"):
            self.assertEqual(gates[name]["status"], "failed")
        self.assertEqual(gates["cross_seed_frequency_mae"]["measured"], .08)

    def test_mismatched_or_nonfinite_evidence_is_rejected(self):
        mutations = [
            lambda r,f,c: r[1]["config"].update(seed=7201),
            lambda r,f,c: r[1]["config"].update(depth_bb=40),
            lambda r,f,c: f[0].update(policy_sha256="0"*64),
            lambda r,f,c: f[0].update(comparison_sha256="0"*64),
            lambda r,f,c: f[0].update(comparison_action_frequency_mae=float("nan")),
            lambda r,f,c: f[0]["baseline"].update(mean_house_rake_bb=.4),
            lambda r,f,c: c[0]["game"].update(small_blind_bb=.5),
            lambda r,f,c: c[0].update(sample_mean_total_nash_conv_bb=1.),
            lambda r,f,c: c[0]["samples"][0][0].update(own_gain_bb=0.),
            lambda r,f,c: c[0].update(abstract_game_optimistic_upper_bound_bb=.5),
            lambda r,f,c: c[0].update(seed=933),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                args = fixtures(); mutate(*args)
                with self.assertRaises(ValueError): audit(*args)

    def test_only_complete_measured_rules_pinned_sources_can_be_loaded(self):
        rules = profile_rules("nl25")
        config = dict(cash_profile="nl25", seed=7201, depth_bb=20, traversals_per_round=32,
                      cash_rules_sha256=rules_digest(rules), cash_network_schema=CASH_NETWORK_SCHEMA,
                      cash_state_feature_schema=CASH_STATE_FEATURE_SCHEMA, cash_terminal_action_integration="own-expectation-v1")
        model = dict(schema=CASH_NETWORK_SCHEMA, strategy_transform="softmax", cash_rules=rules,
                     cash_depth_bb=20, input_size=CASH_STATE_FEATURE_COUNT+9, cash_action_abstraction={"grid":"test"})
        raw = json.dumps(model).encode()
        source = dict(validation_status="training_not_activated", config=config, round=1, payoff_contract=PAYOFF_CONTRACT,
                      rules_sha256=rules_digest(rules), cash_rules=rules, native_average_policy_sha256=hashlib.sha256(raw).hexdigest())
        state = dict(config=config, config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",", ":")).encode()).hexdigest(),
                     completed_rounds=1, completed_traversals=32,
                     metrics=[dict(round=1, elapsed_seconds=1., records_truncated=False)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); artifact = root / "artifacts/round-000001"; artifact.mkdir(parents=True)
            (root/"state.json").write_text(json.dumps(state))
            (artifact/"model-source.json").write_text(json.dumps(source))
            (artifact/"average-policy.json").write_bytes(raw)
            self.assertEqual(load_run(root)["training_hours"], 1/3600)
            (artifact/"average-policy.json").write_bytes(raw+b" ")
            with self.assertRaises(ValueError): load_run(root)
            (artifact/"average-policy.json").write_bytes(raw)
            state["metrics"] = []
            (root/"state.json").write_text(json.dumps(state))
            with self.assertRaises(ValueError): load_run(root)


if __name__ == "__main__": unittest.main()
