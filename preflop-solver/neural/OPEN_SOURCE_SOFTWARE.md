# Neural training software

The neural training path is intentionally usable without paid services or
proprietary training frameworks.

| Component | Use | License | Upstream |
| --- | --- | --- | --- |
| MLX | Apple-silicon tensor training | MIT | https://github.com/ml-explore/mlx |
| NumPy | bounded memory-mapped replay arrays | BSD-3-Clause | https://github.com/numpy/numpy |
| OpenSpiel | optional algorithm/conformance reference | Apache-2.0 | https://github.com/google-deepmind/open_spiel |
| PyTorch | optional future NVIDIA/MPS backend, not required | BSD-style | https://github.com/pytorch/pytorch |
| flate2 and transitive miniz crates | compact Rust trajectory output | MIT OR Apache-2.0 | https://github.com/rust-lang/flate2-rs |

OpenSpiel and PyTorch are not runtime dependencies. Frozen browser artifacts
contain only project metadata and framework-neutral `float32` parameters.

The bootstrapped cumulative-advantage update is implemented independently from
the published [Deep (Predictive) Discounted Counterfactual Regret Minimization
paper](https://arxiv.org/abs/2511.08174). The authors' accompanying research
repository is not vendored or used as a dependency because it did not include
an explicit open-source license when evaluated.
