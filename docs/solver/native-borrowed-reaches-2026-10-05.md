# Native traversal range-copy experiment

Date: October 5, 2026. Status: exact parity passed; cost inconclusive;
implementation not retained. This is not a policy-quality intervention.

A two-second sample of a native64 worker showed training traversal,
compatible-mass kernels, vector movement and strategy generation. This short
sample is not a whole-run percentage or proof of a memory-bandwidth bottleneck.

The isolated proposal copies only the acting player's 1326-entry f64 range at
betting edges and borrows the unchanged opponent range. Chance still masks both
ranges. Owned entry points, update order, discounting, arithmetic, legal actions
and exports stay unchanged. Halving bytes copied at that edge does not imply
halving total allocation, process memory or runtime.

## Verification and decision

- All 64 targeted native tests pass, including input immutability and differential
  checks of regrets, averages, clocks, exports and responses.
- `cargo test --release -- --test-threads=2`: 363 unit and nine CLI tests pass;
  57 research jobs remain ignored. Tests ran under concurrent load.
- Native64 limped turn0 packet is byte-identical, SHA-256
  `25a6bd363f1edd8b076919299287827dca8f525ff407c9797e1ecd3ba39ff0cd`.
- Default learned128 also reproduces candidate
  `90b41a1ee7177663a0aa55d62438925414201f84dd25958a87926e550c65ba83`.

After competing solver work finished, a guarded serial old/new/old replay took
58.251s / 59.507s / 70.173s. All packets are byte-identical. The old controls
drifted 18.57%, exceeding the predeclared 15% stability limit. The nominal 7.33%
reduction against their mean is therefore inconclusive, not a useful speed gain.
Peak memory varied 428,212,992 / 606,241,536 / 727,401,312 bytes; these observations
also do not establish a memory improvement. System memory pressure stayed normal.

Do not retain or push this source optimization on these results. Its forward
patch is preserved as a diagnostic artifact; only our isolated edits were
reversed. The earlier bounded streaming-hash optimization remains retained
because its exact-parity memory improvement was demonstrated separately.

Artifacts: `preflop-solver/neural/runs/local-borrowed-reaches-20261005-a/`
contains `proposal.patch`, `build.json`, `packet-preflight/manifest.json`,
`default-preflight/manifest.json` and `serial-cost/manifest.json`.
Experimental binary:
`a6d3b532c0c6f37ca2c863f23524071c9f59e518bddde877a514679cf6734c8e`.
No neural weights, website behavior or GTO qualification changed.
