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
    BLOCKER_POOLED_CONTRACT, BASELINE_CONDITIONED_NETWORK_SCHEMA, BASELINE_CONDITIONED_CONTRACT,
    POT_EXPERT_NETWORK_SCHEMA, POT_EXPERT_CONTRACT, KERNEL_POOLED_NETWORK_SCHEMA, KERNEL_POOLED_CONTRACT)
from cash_range_pooling import numpy_card_removed_opponent_pool
from cash_range_attention import numpy_positive_kernel_opponent_pool
from train_public_value_network import RANGE_POOL_EPSILON
from cash_frozen_adapter import validate_frozen_adapter, frozen_adapter_coefficients
from validate_public_value_parity import dense_forward
from serving_value_projection import OwnPayoffValueProjection


def cash_architecture(model):
    adapter = validate_frozen_adapter(model)
    if adapter is not None:
        parent = dict(model, schema=KERNEL_POOLED_NETWORK_SCHEMA,
                      architecture="wide-kernel-pooled", predictionContract=KERNEL_POOLED_CONTRACT)
        del parent["frozenTailAdapter"]
        return cash_architecture(parent)
    architecture = model.get("architecture")
    if model.get("schema") == KERNEL_POOLED_NETWORK_SCHEMA:
        for name in ("kernelQuery","kernelKey"):
            layers = model.get(name)
            if (not isinstance(layers,list) or len(layers)!=1 or not isinstance(layers[0],dict)
                    or type(layers[0].get("inputSize")) is not int or layers[0]["inputSize"]!=64
                    or type(layers[0].get("outputSize")) is not int or layers[0]["outputSize"]!=8
                    or layers[0].get("activation")!="linear" or layers[0].get("normalization") is not None
                    or layers[0].get("residual",False) is not False
                    or layers[0].get("normalizationWeights",[]) or layers[0].get("normalizationBiases",[])):
                raise ValueError("cash kernel parity requires two plain64-to8 linear projections")
    elif (model.get("kernelQuery",[]) or model.get("kernelKey",[])
          or model.get("kernelRank") is not None or model.get("kernelFeatureMap") is not None):
        raise ValueError("cash parity cannot silently ignore a kernel projection")
    if ((model.get("schema") == NETWORK_SCHEMA and architecture in ("compact", "wide"))
            or (model.get("schema") == POOLED_NETWORK_SCHEMA and architecture == "wide-pooled")
            or (model.get("schema") == BLOCKER_POOLED_NETWORK_SCHEMA and architecture == "wide-blocker-pooled"
                and model.get("predictionContract") == BLOCKER_POOLED_CONTRACT)
            or (model.get("schema") == BASELINE_CONDITIONED_NETWORK_SCHEMA and architecture == "wide-baseline-conditioned"
                and model.get("predictionContract") == BASELINE_CONDITIONED_CONTRACT)
            or (model.get("schema") == POT_EXPERT_NETWORK_SCHEMA and architecture == "wide-pooled-pot-experts"
                and model.get("predictionContract") == POT_EXPERT_CONTRACT
                and model.get("head") == [] and isinstance(model.get("potExpertHeads"),list)
                and len(model["potExpertHeads"]) == 3 and all(model["potExpertHeads"]))
            or (model.get("schema") == KERNEL_POOLED_NETWORK_SCHEMA and architecture == "wide-kernel-pooled"
                and model.get("predictionContract") == KERNEL_POOLED_CONTRACT and model.get("kernelRank") == 8
                and model.get("kernelFeatureMap") == "elu-plus-one-v1")):
        return architecture
    raise ValueError("cash parity requires matching versioned architecture and own-payoff schema")


def model_features(source, model):
    features = cash_feature_schema(model)
    normalization = model.get("valueNormalization", "pot")
    if normalization == "pot":
        return feature_arrays(source,features)
    return feature_arrays(source,features,value_normalization=normalization)


def python_predictions(source, model):
    architecture = cash_architecture(model)
    contexts, queries, weights, scales, baselines, legal, _, _ = model_features(source,model)
    embeddings = dense_forward(contexts,model["contextTower"])
    combo_embeddings = dense_forward(queries,model["queryTower"])
    expanded = np.broadcast_to(embeddings[:,:,None,:],combo_embeddings.shape)
    if architecture in ("wide-pooled","wide-blocker-pooled","wide-baseline-conditioned","wide-pooled-pot-experts","wide-kernel-pooled"):
        reach = weights / np.maximum(weights.sum(axis=2,keepdims=True),RANGE_POOL_EPSILON)
        pooled = np.sum(combo_embeddings * reach[:,:,:,None],axis=2)
        own = np.broadcast_to(pooled[:,:,None,:],combo_embeddings.shape)
        if architecture == "wide-kernel-pooled":
            def positive(value): return np.where(value>=0,value+1,np.exp(np.minimum(value,0)))
            opponent = numpy_positive_kernel_opponent_pool(combo_embeddings,
                np.asarray([label["input"]["state"]["ranges"] for label in source["labels"]],dtype=np.float32),
                positive(dense_forward(combo_embeddings,model["kernelQuery"])),
                positive(dense_forward(combo_embeddings,model["kernelKey"])))
        else:
            opponent = (numpy_card_removed_opponent_pool(combo_embeddings,np.asarray([label["input"]["state"]["ranges"] for label in source["labels"]],dtype=np.float32))
                        if architecture == "wide-blocker-pooled" else np.broadcast_to(pooled[:,::-1,None,:],combo_embeddings.shape))
        combined = np.concatenate((expanded,own,opponent,combo_embeddings),axis=-1)
    else:
        combined = np.concatenate((expanded,combo_embeddings),axis=-1)
    if architecture == "wide-baseline-conditioned":
        combined = np.concatenate((combined,baselines[:,:,:,None]/20.),axis=-1)
    if architecture == "wide-pooled-pot-experts":
        invested = np.maximum(contexts[:,0,19],contexts[:,0,20])*20
        indices = np.where(invested<=3.5,0,np.where(invested<=7.5,1,2))
        residual = np.empty((len(contexts),2,1326),np.float32)
        for index,head in enumerate(model["potExpertHeads"]):
            selected = indices == index
            if selected.any():
                residual[selected] = dense_forward(combined[selected],head).reshape((-1,2,1326))
    else:
        residual = dense_forward(combined,model["head"]).reshape((-1,2,1326))
    raw = baselines + residual * scales[:,None,None]
    adapter = validate_frozen_adapter(model)
    if adapter is not None:
        coefficients = frozen_adapter_coefficients(source, adapter)
        active = coefficients > 0
        if active.any():
            raw[active] += dense_forward(combined[active], adapter["head"]).reshape((-1,2,1326)) * coefficients[active,None,None]
    return OwnPayoffValueProjection(raw,legal.astype(bool)).values


def mlx_predictions(source, payload, device):
    with mx.stream(device):
        features = cash_feature_schema(payload)
        model = OwnComboValueNetwork(cash_architecture(payload),features,value_normalization=payload.get("valueNormalization","pot"))
        towers = [("context_tower",payload["contextTower"]),("query_tower",payload["queryTower"])]
        if model.uses_pot_experts:
            towers.extend((f"pot_expert_heads.{index}",head) for index,head in enumerate(payload["potExpertHeads"]))
        else:
            towers.append(("head",payload["head"]))
        if model.architecture == "wide-kernel-pooled":
            towers.extend((("kernel_query",payload["kernelQuery"]),("kernel_key",payload["kernelKey"])))
        weights = []
        for tower, layers in towers:
            for index, layer in enumerate(layers):
                weights.extend([(f"{tower}.layers.{index*2}.weight", mx.array(np.asarray(layer["weights"],dtype=np.float32).reshape(layer["outputSize"],layer["inputSize"]))),
                                (f"{tower}.layers.{index*2}.bias", mx.array(layer["biases"]))])
        model.load_weights(weights)
        contexts, queries, weights, scales, baselines, legal, _, _ = model_features(source,payload)
        arrays = [mx.array(v) for v in (contexts,queries,weights,scales,baselines,legal)]
        if model.architecture in ("wide-blocker-pooled","wide-kernel-pooled"):
            arrays.append(mx.array(np.asarray([label["input"]["state"]["ranges"] for label in source["labels"]],dtype=np.float32)))
        adapter = validate_frozen_adapter(payload)
        if adapter is None:
            return np.array(model(*arrays)).reshape((-1,2,1326)) * scales[:,None,None]
        parent = np.array(model(*arrays)).reshape((-1,2,1326)) * scales[:,None,None]
        coefficients = frozen_adapter_coefficients(source, adapter)
        active = coefficients > 0
        if not active.any():
            return parent
        # Use the same frozen encoder; only the independent extra head is new.
        context_embedding = model.context_tower(arrays[0])
        hands = model.query_tower(arrays[1])
        reach = arrays[2] / mx.maximum(mx.sum(arrays[2],axis=2,keepdims=True),RANGE_POOL_EPSILON)
        own = mx.sum(hands*reach[:,:,:,None],axis=2)
        positive = lambda value: mx.where(value>=0,value+1,mx.exp(mx.minimum(value,0)))
        from cash_range_attention import positive_kernel_opponent_pool
        opponent = positive_kernel_opponent_pool(hands,arrays[6],
            positive(model.kernel_query(hands)),positive(model.kernel_key(hands)))
        combined = mx.concatenate((mx.broadcast_to(context_embedding[:,:,None,:],hands.shape),
            mx.broadcast_to(own[:,:,None,:],hands.shape),opponent,hands),axis=-1)
        delta = combined
        for layer in adapter["head"]:
            weight = mx.array(np.asarray(layer["weights"],np.float32).reshape(layer["outputSize"],layer["inputSize"]))
            delta = delta @ weight.T + mx.array(layer["biases"])
            if layer["activation"] == "relu": delta = mx.maximum(delta,0.)
        raw = model.raw_values(*arrays[:5],ranges=arrays[6]) * arrays[3][:,None,None]
        corrected = mx.clip(raw + delta.reshape((-1,2,1326))*mx.array(coefficients)[:,None,None],-20.,20.) * arrays[5][:,None,:]
        return np.array(mx.where(mx.array(active)[:,None,None],corrected,mx.array(parent)))


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
