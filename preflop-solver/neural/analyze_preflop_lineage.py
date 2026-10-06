"""Read-only root audit separating teacher, distillation and serving errors.

Commercial chart frequencies are a diagnostic reference, never training targets
or an exploitability certificate. EVs from different continuation games are
reported side by side, never scored as if they shared one payoff function.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

RANKS = "23456789TJQKA"
ROOT_HISTORY = ["blinds:0.500/1.000"]
CLASSES = {
    RANKS[hi] + RANKS[lo] + suffix
    for hi in range(13)
    for lo in range(hi + 1)
    for suffix in ([""] if hi == lo else ["s", "o"])
}
GROUPS = ("fold", "limp", "raise", "shove")


def multiplicity(hand):
    if hand not in CLASSES:
        raise ValueError(f"invalid canonical hand class: {hand}")
    return 6 if len(hand) == 2 else 4 if hand.endswith("s") else 12


def probabilities(values, count):
    if (not isinstance(values, list) or len(values) != count or count == 0
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   or v < 0 or v > 1 for v in values)
            or abs(math.fsum(values) - 1) > 1e-9):
        raise ValueError("invalid probability vector")
    return values


def indexed_roots(rows, value_rows=False):
    output = {}
    for row in rows:
        actor = row.get("player" if value_rows else "actor")
        if actor != 0 or row.get("public_history") != ROOT_HISTORY:
            continue
        hand = row["hand_class"]
        if hand in output:
            raise ValueError("duplicate SB root hand class")
        multiplicity(hand)
        if row.get("key") != f"p0|{hand}|blinds:0.500/1.000":
            raise ValueError("root key, hand and public history disagree")
        labels = row["action_labels"]
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("invalid root action labels")
        probabilities(row["policy_probabilities" if value_rows else "probabilities"], len(labels))
        output[hand] = row
    if set(output) != CLASSES:
        raise ValueError("SB root requires exactly all 169 hand classes")
    first_labels = next(iter(output.values()))["action_labels"]
    if any(row["action_labels"] != first_labels for row in output.values()):
        raise ValueError("root legal action order differs across hands")
    return output


def policy_roots(artifact):
    if artifact.get("schema") != "hu-tabular-preflop-dcfr-v1":
        raise ValueError("incompatible frozen preflop policy")
    game = artifact["game"]
    if (artifact.get("depth_bb") != 20 or game.get("effective_stack_bb") != 20
            or game.get("small_blind_bb") != 0.5 or game.get("big_blind_bb") != 1):
        raise ValueError("audit requires matching 20bb and 0.5/1 blinds")
    return indexed_roots(artifact["strategies"])


def root_comparison(first, second):
    mae = tv = agreement = maximum = 0.0
    for hand in sorted(CLASSES):
        a, b = first[hand], second[hand]
        if a["action_labels"] != b["action_labels"]:
            raise ValueError("comparison requires identical legal actions and order")
        p, q = a["probabilities"], b["probabilities"]
        deltas = [abs(x - y) for x, y in zip(p, q)]
        weight = multiplicity(hand) / 1326
        mae += weight * math.fsum(deltas) / len(deltas)
        tv += weight * math.fsum(deltas) / 2
        agreement += weight * (p.index(max(p)) == q.index(max(q)))
        maximum = max(maximum, *deltas)
    return dict(comboWeightedActionMae=mae, comboWeightedTotalVariation=tv,
                comboWeightedPrimaryAgreement=agreement, maximumActionDelta=maximum)


def group_action(label):
    if label == "fold":
        return "fold"
    if label == "limp":
        return "limp"
    if label.startswith("raise_all_in_to_"):
        return "shove"
    if label.startswith("raise_to_"):
        return "raise"
    raise ValueError(f"unrecognized SB opening action: {label}")


def aggregate(roots):
    totals = {group: [] for group in GROUPS}
    for hand, row in roots.items():
        weight = multiplicity(hand) / 1326
        for label, probability in zip(row["action_labels"], row["probabilities"]):
            totals[group_action(label)].append(weight * probability)
    return {group: math.fsum(parts) for group, parts in totals.items()}


def served_roots(receipt, student):
    if receipt.get("schema") != "served-preflop-range-audit-v1":
        raise ValueError("incompatible native query audit")
    if receipt.get("assumptions") != dict(depthBb=20, blindsBb=[0.5, 1],
                                          anteBb=0, rake="none", cardCombos=1326):
        raise ValueError("native query audit has different game assumptions")
    matches = [r for r in receipt["summaries"] if r["context"] == "sb-first-in"]
    if len(matches) != 1 or matches[0]["queriedCombos"] != 1326:
        raise ValueError("native audit must query all 1,326 root combinations")
    context = matches[0]
    if context["maximumWithinClassProbabilitySpread"] > 1e-12:
        raise ValueError("native audit is not suit invariant; do not collapse to classes")
    labels = next(iter(student.values()))["action_labels"]
    transport = ["fold:" if label == "fold" else "call:" if label == "limp"
                 else ("all_in:" if group_action(label) == "shove" else "raise:")
                 + format(float(label.split("_to_")[1][:-2]), ".15g")
                 for label in labels]
    if context["actionNames"] != transport:
        raise ValueError("native and student legal actions differ")
    rows = []
    for row in context["hands"]:
        if row["combos"] != multiplicity(row["hand"]):
            raise ValueError("native hand class has incorrect combo multiplicity")
        rows.append(dict(key=f"p0|{row['hand']}|blinds:0.500/1.000", actor=0,
                         hand_class=row["hand"], public_history=ROOT_HISTORY,
                         action_labels=labels, probabilities=row["probabilities"]))
    return indexed_roots(rows)


def value_comparison(old_artifact, new_artifact, teacher, student):
    old = indexed_roots([r for player in old_artifact["players"] for r in player], True)
    new = indexed_roots([r for player in new_artifact["players"] for r in player], True)
    counts = [{group: 0 for group in GROUPS} for _ in range(2)]
    flipped = []
    examples = []
    for hand in sorted(CLASSES):
        for rows, policy, histogram in zip((old, new), (teacher, student), counts):
            row = rows[hand]
            if row["action_labels"] != policy[hand]["action_labels"]:
                raise ValueError("action EV labels differ from their own source policy")
            if max(abs(p - q) for p, q in zip(row["policy_probabilities"],
                                           policy[hand]["probabilities"])) > 1e-9:
                raise ValueError("action EV probabilities differ from their own source policy")
            evs = row["action_values_bb"]
            if len(evs) != len(row["action_labels"]) or any(not math.isfinite(v) for v in evs):
                raise ValueError("invalid root action EVs")
            histogram[group_action(row["action_labels"][evs.index(max(evs))])] += 1
        if old[hand]["action_labels"] != new[hand]["action_labels"]:
            raise ValueError("old and new EVs have different action menus")
        old_evs, new_evs = old[hand]["action_values_bb"], new[hand]["action_values_bb"]
        labels = old[hand]["action_labels"]
        if labels[old_evs.index(max(old_evs))] == "fold" and labels[new_evs.index(max(new_evs))] != "fold":
            flipped.append(hand)
        if hand in {"AA", "KK", "AKs", "K2o", "76s", "72o", "32o"}:
            examples.append(dict(hand=hand, actionLabels=labels, oldTeacherEvsBb=old_evs,
                                 newerStudentEvsBb=new_evs))
    return dict(oldBestActionClassCounts=counts[0], newerBestActionClassCounts=counts[1],
                oldFoldNowContinueClasses=len(flipped),
                oldFoldNowContinueComboFraction=sum(multiplicity(h) for h in flipped)/1326,
                examples=examples, identicalContinuationGameEstablished=False,
                interpretation="Different policy profiles and continuation generators: EV shifts localize target drift, not an improvement or a matched exploitability comparison.")


def continuation_route_comparison(provenance, runtime):
    """The v2 range-cache generator re-solves both seats, without a blueprint.

    This is intentionally a fail-closed diagnostic, not runtime validation:
    experimental serving still discloses uncalibrated continuation uncertainty.
    Matching iteration counts alone does not make these payoff functions equal.
    """
    solver = runtime["resolver"]
    dcfr = runtime["dcfr"]
    checks = dict(
        iterationsMatch=provenance["iterations"] == solver["flopIterations"],
        actorRoutingMatch=solver["flopResolvedActor"] is None,
        dcfrMatch=all(provenance["dcfr"][key] == dcfr[name] for key, name in (
            ("positive_regret_exponent", "positiveRegretExponent"),
            ("negative_regret_exponent", "negativeRegretExponent"),
            ("strategy_exponent", "strategyExponent"))),
        deployedPostflopBlueprintEvaluated=False,
        laterStreetRoutingEvaluated=False,
    )
    return dict(checks=checks, canUseAsMatchedServingEvaluation=all(checks.values()),
                interpretation="v2 canonical continuation caches evaluate a bilateral flop subgame with learned turn leaves. They do not roll out the deployed full-hand policy; small chance SE is not a calibration bound.")


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("student-policy", "served-audit", "teacher-action-values", "student-action-values",
                 "student-networks", "ancestor-networks", "continuation-shard", "manifest", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--teacher-policy", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite immutable audit output")
    inputs = {str(p): digest(p) for value in vars(args).values()
              for p in (value if isinstance(value, list) else [value]) if p != args.output}
    student_artifact = read(args.student_policy)
    student = policy_roots(student_artifact)
    receipt = read(args.served_audit)
    native = served_roots(receipt, student)
    parity = root_comparison(student, native)
    if parity["maximumActionDelta"] > 1e-12:
        raise ValueError("native serving differs from archived student export")
    teachers = [read(p) for p in args.teacher_policy]
    teacher_roots = [policy_roots(t) for t in teachers]
    game_fields = ("effective_stack_bb", "small_blind_bb", "big_blind_bb", "action_abstraction")
    if any(any(t["game"][key] != student_artifact["game"][key] for key in game_fields) for t in teachers):
        raise ValueError("teacher and student abstract games differ")
    networks, ancestor = read(args.student_networks), read(args.ancestor_networks)
    if (networks["networks"] != ancestor["networks"]
            or networks["input_size"] != ancestor["input_size"]
            or networks["strategy_transform"] != ancestor["strategy_transform"]):
        raise ValueError("served preflop networks differ from declared ancestor")
    network_bytes = args.student_networks.read_bytes()
    if args.student_networks.suffix == ".gz":
        network_bytes = gzip.decompress(network_bytes)
    network_sha256 = hashlib.sha256(network_bytes).hexdigest()
    if student_artifact["source_policy_sha256"] != network_sha256:
        raise ValueError("archived student policy targets different network bytes")
    reference = read(args.reference)
    if reference["depthBb"] != 20 or reference["anteBb"] != 0 or reference["groups"] != list(GROUPS):
        raise ValueError("incompatible external diagnostic reference")
    ref = probabilities(reference["probabilities"], len(GROUPS))
    mix = aggregate(student)
    continuation = read(args.continuation_shard)
    new_values = read(args.student_action_values)
    old_values = read(args.teacher_action_values)
    if (continuation["schema"] != "hu-preflop-range-continuation-cache-v2"
            or continuation["policy_sha256"] != new_values["policy_artifact_sha256"]
            or new_values["policy_artifact_sha256"] != digest(args.student_policy)
            or new_values["source_policy_sha256"] != student_artifact["source_policy_sha256"]
            or old_values["policy_model_version"] != teachers[0]["model_version"]):
        raise ValueError("continuation/action-value/policy provenance mismatch")
    manifests = [m for m in read(args.manifest) if m["version"] == receipt["modelVersion"]]
    if len(manifests) != 1:
        raise ValueError("native audit must match exactly one pinned manifest")
    if manifests[0]["runtime"]["networkSha256"] != network_sha256:
        raise ValueError("served manifest targets different network bytes")
    result = dict(schema="hu-preflop-lineage-audit-v1", inputSha256=inputs,
        modelVersion=receipt["modelVersion"], nativeBinarySha256=receipt["binarySha256"],
        completeRootClasses=169, exactRootCombos=1326, servingParity=parity,
        preflopAncestorWeightsIdentical=True, studentOpeningMix=mix,
        teacherComparisons=[dict(modelVersion=t["model_version"], openingMix=aggregate(r),
                                examples=[dict(hand=h, probabilities=r[h]["probabilities"])
                                          for h in ("AA", "KK", "AKs", "K2o", "76s")],
                                **root_comparison(r, student)) for t, r in zip(teachers, teacher_roots)],
        reference=dict(source=reference["source"], actionTreeIdentical=False,
            referenceOpeningMix=dict(zip(GROUPS, ref)),
            aggregateDelta={g: mix[g]-p for g, p in zip(GROUPS, ref)},
            aggregateTotalVariation=math.fsum(abs(mix[g]-p) for g,p in zip(GROUPS,ref))/2),
        values=value_comparison(old_values, new_values, teacher_roots[0], student),
        newerContinuationProvenance=continuation["resolver_provenance"],
        continuationRouteComparison=continuation_route_comparison(continuation["resolver_provenance"],
                                                                  manifests[0]["runtime"]),
        newerContinuationMethod="bilateral flop re-solve with learned turn values, not a rollout of the deployed BB-only routed policy",
        releaseAccepted=False, activeModelChanged=False,
        interpretation="Root-only lineage diagnostic. Commercial action-frequency differences and internal teacher fit are not exploitability or equilibrium certificates.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves the receipt and fails closed on concurrent reruns.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), outputSha256=digest(args.output),
                         studentOpeningMix=mix, teacherOpeningMixes=[aggregate(r) for r in teacher_roots],
                         servingMaximumDelta=parity["maximumActionDelta"],
                         newerBestActionCounts=result["values"]["newerBestActionClassCounts"],
                         releaseAccepted=False), indent=2))


if __name__ == "__main__":
    main()
