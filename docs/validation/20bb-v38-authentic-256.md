# 20bb v38 authentic-corpus expansion

Status: rejected; no active manifest was modified.

The exact range-conditioned turn corpus was expanded from 128 to 256
authentic frozen-policy states. Generation completed from immutable per-state
checkpoints with source-policy SHA-256
`c78397af5900b3409d3dfc911fce56075cb54ce860c38fc2a1459fe5d56df948`.
The assembled dataset SHA-256 is
`d75cb33e34ce023bafb57ed3a9daa99052b0b1f67b8d8261d24ce464533eeb84`.

All source gates passed: maximum belief-replicate total variation was
`0.128834`, maximum exact river abstraction exploitability was
`0.035762bb/hand`, and minimum effective sample size was `1216.43` of 4096.
The 64-state holdout was drawn only from newly generated indices 128--255.

## Controlled fits

The compact fit used only authentic states. The supplemented fit restored the
v32 12-state off-policy and v33 18-state resolver-leaf training corpora with a
50% authentic-primary batch floor. Both used the same split seed, 3,000 steps,
batch size 6, pot normalization, and two independent model seeds.

| Fit | Paired holdout RMSE | Small | Medium | Large |
| --- | ---: | ---: | ---: | ---: |
| Pure authentic compact | 0.678345bb | 0.177488bb | 0.551001bb | 1.453788bb |
| Supplemented compact | 0.606540bb | 0.194957bb | 0.546952bb | 1.265483bb |
| Prior v36 pair on the same states | 0.596164bb | 0.204014bb | 0.607770bb | 1.211659bb |

The supplements recovered most of the pure-corpus regression but remained
1.74% worse than v36 overall. Both fits fail the absolute `0.25bb` authentic
RMSE ceiling. Their large-pot error remains the dominant contribution, so no
matched resolver compute or model promotion was authorized.

## Split defect discovered downstream

The originally frozen split was deterministic and disjoint, but not
pot-stratified. It put 12 large-pot states into validation and zero into the
26-state tuning set. That made early stopping blind to the largest measured
error regime. v38 remains a valid rejected measurement, but it must not be
used to select architecture or stopping time. v39 corrects the splitter and
reports each split's pot-band counts explicitly.
