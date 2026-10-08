#!/usr/bin/env python3
"""Deterministic frozen-weight cash parity loop; no fitting required."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import mlx.core as mx
from train_cash_value_network import OwnComboValueNetwork
from train_cash_value_network import feature_arrays
from validate_public_value_parity import dense_forward
from serving_value_projection import OwnPayoffValueProjection


def python_predictions(source, model):
    contexts, queries, weights, scales, baselines, legal, _, _ = feature_arrays(source)
    embeddings = dense_forward(contexts,model["contextTower"])
    combo_embeddings = dense_forward(queries,model["queryTower"])
    combined = np.concatenate((np.broadcast_to(embeddings[:,:,None,:],combo_embeddings.shape),combo_embeddings),axis=-1)
    residual = dense_forward(combined,model["head"]).reshape((-1,2,1326))
    return OwnPayoffValueProjection(baselines + residual * scales[:,None,None],legal.astype(bool)).values


def mlx_predictions(source, payload, device):
    with mx.stream(device):
        model = OwnComboValueNetwork()
        names = {"context_tower":"contextTower", "query_tower":"queryTower", "head":"head"}
        weights = []
        for tower, name in names.items():
            for index, layer in enumerate(payload[name]):
                weights.extend([(f"{tower}.layers.{index*2}.weight", mx.array(np.asarray(layer["weights"],dtype=np.float32).reshape(layer["outputSize"],layer["inputSize"]))),
                                (f"{tower}.layers.{index*2}.bias", mx.array(layer["biases"]))])
        model.load_weights(weights)
        contexts, queries, weights, scales, baselines, legal, _, _ = feature_arrays(source)
        arrays = [mx.array(v) for v in (contexts,queries,weights,scales,baselines,legal)]
        return np.array(model(*arrays)).reshape((-1,2,1326)) * scales[:,None,None]


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
