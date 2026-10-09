#!/usr/bin/env python3
"""Deterministic frozen-weight cash parity loop; no fitting required."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import mlx.core as mx
from train_cash_value_network import OwnComboValueNetwork, CASH_FEATURE_SCHEMAS
from train_cash_value_network import feature_arrays
from cash_value_dataset import (NETWORK_SCHEMA, POOLED_NETWORK_SCHEMA, BLOCKER_POOLED_NETWORK_SCHEMA,
    BLOCKER_POOLED_CONTRACT, BASELINE_CONDITIONED_NETWORK_SCHEMA, BASELINE_CONDITIONED_CONTRACT)
from cash_range_pooling import numpy_card_removed_opponent_pool
from train_public_value_network import RANGE_POOL_EPSILON
from validate_public_value_parity import dense_forward
from serving_value_projection import OwnPayoffValueProjection


def cash_architecture(model):
    architecture = model.get("architecture")
    if ((model.get("schema") == NETWORK_SCHEMA and architecture in ("compact", "wide"))
            or (model.get("schema") == POOLED_NETWORK_SCHEMA and architecture == "wide-pooled")
            or (model.get("schema") == BLOCKER_POOLED_NETWORK_SCHEMA and architecture == "wide-blocker-pooled"
                and model.get("predictionContract") == BLOCKER_POOLED_CONTRACT)
            or (model.get("schema") == BASELINE_CONDITIONED_NETWORK_SCHEMA and architecture == "wide-baseline-conditioned"
                and model.get("predictionContract") == BASELINE_CONDITIONED_CONTRACT)):
        return architecture
    raise ValueError("cash parity requires matching versioned architecture and own-payoff schema")


def python_predictions(source, model):
    architecture = cash_architecture(model)
    features = cash_feature_schema(model)
    contexts, queries, weights, scales, baselines, legal, _, _ = feature_arrays(source,features)
    embeddings = dense_forward(contexts,model["contextTower"])
    combo_embeddings = dense_forward(queries,model["queryTower"])
    expanded = np.broadcast_to(embeddings[:,:,None,:],combo_embeddings.shape)
    if architecture in ("wide-pooled","wide-blocker-pooled","wide-baseline-conditioned"):
        reach = weights / np.maximum(weights.sum(axis=2,keepdims=True),RANGE_POOL_EPSILON)
        pooled = np.sum(combo_embeddings * reach[:,:,:,None],axis=2)
        own = np.broadcast_to(pooled[:,:,None,:],combo_embeddings.shape)
        opponent = (numpy_card_removed_opponent_pool(combo_embeddings,np.asarray([label["input"]["state"]["ranges"] for label in source["labels"]],dtype=np.float32))
                    if architecture == "wide-blocker-pooled" else np.broadcast_to(pooled[:,::-1,None,:],combo_embeddings.shape))
        combined = np.concatenate((expanded,own,opponent,combo_embeddings),axis=-1)
    else:
        combined = np.concatenate((expanded,combo_embeddings),axis=-1)
    if architecture == "wide-baseline-conditioned":
        combined = np.concatenate((combined,baselines[:,:,:,None]/20.),axis=-1)
    residual = dense_forward(combined,model["head"]).reshape((-1,2,1326))
    return OwnPayoffValueProjection(baselines + residual * scales[:,None,None],legal.astype(bool)).values


def mlx_predictions(source, payload, device):
    with mx.stream(device):
        features = cash_feature_schema(payload)
        model = OwnComboValueNetwork(cash_architecture(payload),features)
        names = {"context_tower":"contextTower", "query_tower":"queryTower", "head":"head"}
        weights = []
        for tower, name in names.items():
            for index, layer in enumerate(payload[name]):
                weights.extend([(f"{tower}.layers.{index*2}.weight", mx.array(np.asarray(layer["weights"],dtype=np.float32).reshape(layer["outputSize"],layer["inputSize"]))),
                                (f"{tower}.layers.{index*2}.bias", mx.array(layer["biases"]))])
        model.load_weights(weights)
        contexts, queries, weights, scales, baselines, legal, _, _ = feature_arrays(source,features)
        arrays = [mx.array(v) for v in (contexts,queries,weights,scales,baselines,legal)]
        if model.architecture == "wide-blocker-pooled":
            arrays.append(mx.array(np.asarray([label["input"]["state"]["ranges"] for label in source["labels"]],dtype=np.float32)))
        return np.array(model(*arrays)).reshape((-1,2,1326)) * scales[:,None,None]


def cash_feature_schema(model):
    if model.get("featureSchema") not in CASH_FEATURE_SCHEMAS:
        raise ValueError("cash parity requires an explicit supported feature schema")
    return model["featureSchema"]


def compare(source, model_path: Path, binary: Path, work: Path):
    model = json.loads(model_path.read_text()); expected = python_predictions(source,model)
    rows = []
    for index,label in enumerate(source["labels"]):
        path = work / f"parity-input-{index:04}.json"; path.write_text(json.dumps(label["input"]))
        completed = subprocess.run([str(binary.resolve()),"cash-turn-value-predict","--input",str(path),"--value-network",str(model_path)],capture_output=True,text=True,check=True)
        actual = np.asarray(json.loads(completed.stdout)["counterfactual_values_bb"])
        difference = np.abs(actual-expected[index]); location = np.unravel_index(difference.argmax(),difference.shape)
        rows.append({"state":index,"maximum_difference_bb":float(difference.max()),"location":list(map(int,location)),
                     "native_bb":float(actual[location]),"python_bb":float(expected[index][location])})
    return {"schema":"cash-frozen-value-parity-v1","maximum_difference_bb":max(r["maximum_difference_bb"] for r in rows),"rows":rows}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset",required=True,type=Path); parser.add_argument("--network",required=True,type=Path)
    parser.add_argument("--native-binary",required=True,type=Path); parser.add_argument("--maximum-difference-bb",type=float,default=.001)
    args = parser.parse_args(); source = json.loads(args.dataset.read_text())
    report = compare(source,args.network,args.native_binary,args.network.parent); print(json.dumps(report,indent=2))
    model = json.loads(args.network.read_text()); dense = python_predictions(source,model)
    print(json.dumps({"mlx_cpu_dense_maximum_bb":float(np.abs(mlx_predictions(source,model,mx.cpu)-dense).max()),
                      "mlx_gpu_dense_maximum_bb":float(np.abs(mlx_predictions(source,model,mx.gpu)-dense).max())}))
    if report["maximum_difference_bb"] > args.maximum_difference_bb: raise SystemExit(1)


if __name__ == "__main__": main()
