# Schema-v3 full-game cloud runbook

This runbook launches the corrected trajectory-recall external-sampling DCFR
trainer. It does not launch the older neural `long_run.py` experiment and it
does not merge regret tables from different random seeds.

> **September 4 update: scaling pilot only, production still on hold.** Solver
> `0.3.0` / checkpoint schema **5** adds scale-correct probability handling,
> shared immutable storage, linear cached terminal evaluation, and two opt-in
> variance-reduction modes. The selected accuracy-focused 800-round terminal-
> integration pair reaches 0.407/0.384bb **root local deviation**, not full-game
> exploitability. Use the updated 1,600-round command below with a **45M** cap.
> It is a bounded experiment, not a promise that a $15 run will pass every gate.
> No paid server or automatic server-deletion system has been created.

The [September 4 audit](../validation/2026-09-04-pcs-math-audit.md) is the current
selection record and lists all remaining metrics. Earlier observations below
are historical context, not interchangeable scores: evaluation seeds/counts
and solver versions differ. The new stateless `--opponent-checkdown-baseline`
uses roughly one-third fewer states at 800 rounds, but has worse root values
and primary agreement than integration; keep it as a research comparator.
These two flags are mutually exclusive and require `--public-chance-sampling`.
The runner pins the chosen mode in its fingerprint, summary, and resume checks.

**Start a new schema-5 lineage.** Schema-4 training state cannot resume under
this corrected recurrence. Keep old checkpoints and their original binary if
needed; do not relabel a checkpoint to bypass the schema check. Both JSON-gzip
and MessagePack-gzip are supported codecs for the current schema.

Before any paid launch, confirm the actual provider shape, live hourly price,
taxes, disk/transfer charges, and the user's budget. The solver's information-
set guard and resumable runner do not delete a billed server. Independent
deletion/watchdog verification remains required; an in-server shutdown alone
does not establish that billing stopped. Do not provision from these commands.

The August 25 public-chance vector vertical slice does not lift this hold. Its
exact frozen-update implementation is deterministic and 27--33% faster in
three eight-round fixed-flop pilots, but depth-limited exploitability is worse
on all three textures and remains worse at approximately equal wall time. The
MVP is therefore research-only and is not a cloud-runner option. The next
eligible pilot must add public-board sampling plus compatible-private-combo
enumeration to the full-game abstract trainer while retaining its empirically
stronger alternating traverser schedule.

A follow-up contiguous joint-hand pilot also remains ineligible. Grouping
independent external-sampling lanes by public action is faster than the former
scalar hand loop, and it can vary both traverser and opponent cards. At matched
CPU, however, neither the `2 × 1` nor `1 × 2` batch consistently beat scalar
policy and authentic-coverage diagnostics across both seeds. Do not pass
`--traverser-hand-batch-size` or `--opponent-hand-batch-size` in a paid run. A
future candidate must carry compatible reach vectors directly rather than add
more finite deal lanes, and it must win matched-wall paired gates first.

The compatible-range candidate now traverses the full abstract game. It
samples one public board per alternating round, updates every compatible exact
private hand, and importance-samples one shared opponent action so opponent
nodes do not fan out by private bucket. The cloud runner accepts
`--public-chance-sampling`, pins it in the run fingerprint and resume checks,
and leaves all scalar defaults unchanged. Deterministic MessagePack checkpoint
resume matches uninterrupted public-chance training exactly.

At 400 fixed-DCFR rounds, seeds 26001/26002 reached 6.60M/6.39M information
sets, 0.698/0.778bb root local deviation, and 12.9--13.9% held-out unknown
coverage in about 46 seconds per concurrent worker. That is a large policy and
coverage improvement over the approximate matched-CPU scalar 45k pair, but
cross-seed MAE (10.99%), median TV (30.23%), p95 TV (51.70%), and primary
agreement (50.89%) still fail. These are scaling-pilot results, not promotion
evidence.

The required 800-round local pair then reached 12.48M/11.88M sets with no swap.
Root gain improved to 0.650/0.702bb, max per-action MAE to 7.05%, median/p95 TV
to 24.91%/43.85%, primary agreement to 66.27%, and continuation unknown to
17.45--19.34%. Every major stability measure moved in the right direction, so
a capped 1,600-round fixed-DCFR cloud stage is now justified. The remaining
distance to the gates is still material.

## Capacity choice

The trainer is single-process and deterministic. Parallelism comes from the
two required independent seeds, so two concurrent workers use two CPU cores
while retaining separate memory-heavy tables. Prefer memory capacity and high
single-core speed over a high core count.

- Benchmark an `x2iedn.xlarge` (128 GiB, 4 vCPUs) against an `r7i.4xlarge`
  (128 GiB, 16 vCPUs) with the same deterministic 10k seed before choosing the
  paid host. The former is shaped better for two scalar workers; the latter has
  a newer CPU and leaves cores for future vector traversal. Select on current
  dollars per completed iteration, not hourly price or vCPU count alone.
- Either 128 GiB shape can host the documented 15-million-information-set pair
  under the runner's conservative memory refusal. Keep the terminal 22M-cap
  extension sequential as specified below.
- A 5-million-set pilot pair can start on an `r7i.2xlarge` (64 GiB, 8 vCPUs),
  subject to the runner's live memory preflight.
- Attach at least 150 GiB of encrypted gp3 EBS. Frozen artifacts stream JSON
  through gzip. New checkpoints stream lossless named MessagePack through
  gzip; legacy JSON-gzip checkpoints remain readable. Compression reduces
  persistent size, but it does not reduce live table memory.

AWS's current instance table lists the R7i sizes and memory/vCPU counts in the
[EC2 memory-optimized instance documentation](https://docs.aws.amazon.com/ec2/latest/instancetypes/mo.html).
Confirm the current `us-west-2` on-demand or Spot price immediately before
launch; prices are intentionally not baked into this repository.

The 10k host comparison is infrastructure-only: require identical artifact
hashes for repeated runs on each host and compare iterations/second after the
binary is warm. Do not use a cross-host floating-point artifact mismatch as a
policy selection signal; retain one architecture for both production seeds.
The hard-flop public-belief benchmark is not a proxy for this choice because
that resolver scales across 1/2/4/8 threads, while schema-v3 blueprint training
does not.

The retained 100,000-iteration 20bb scale pilot observed 3,242,828 information
sets and 2.68GB peak RSS (about 32.4 new sets per iteration and 826 live bytes
per set). At 200,000 iterations both matched seeds reached about 6.03 million
sets at 4.99GB peak RSS. At 300,000 iterations the fixed and scheduled pairs
reached 8.61–8.69 million sets at no more than 7.22GB peak RSS. Every run had
zero swaps. The exact selected 400,000-iteration midpoint then reached
11.08–11.17 million sets at about 7.13GB maximum RSS and 9.31GB peak memory
footprint, again without swapping. The earlier part of the curve was similar:
1,662,066 sets at 50,000 iterations. These are sizing observations, not a
promise of linear growth. The runner therefore retains a much more conservative
2,300 bytes per live information set plus 20% headroom and refuses obvious
oversubscription.

The local 600k terminal pair reached 15.93M and 15.81M information sets. On a
16GB laptop, sequential workers peaked at 5.92GB/8.42GB RSS and
13.36GB/13.26GB process footprint; briefly starting both together forced rapid
system swap growth, so the second was stopped and restarted cleanly after the
first. This directly validates the terminal command's `--max-concurrent 1`
choice and the runner's refusal to place a 22M-cap stage on a small host.

The subsequent allocation audit is policy-exact. On the same 20,000-iteration
seed and evaluation controls, the optimized binary reproduced the complete
canonical artifact SHA-256 while reducing peak RSS from 586MB to 377MB. At the
larger guard it reached 2,000,050 information sets at 895MB RSS, about 447 live
bytes per set before full export. A complete-postflop export of 437,080 trained
rows peaked at 787MB and produced a 48MB gzip artifact. These numbers provide
additional headroom but do not weaken the existing 2,300-byte conservative
refusal: export rows, evaluation accumulators, and checkpoint loading can
coexist with the live table.

A 20,000-iteration lossless MessagePack checkpoint was 78MB versus 93MB for
the legacy JSON-gzip form. Its checkpointed run fell from 29.1s to 18.8s and a
same-iteration resume/evaluation from 5.78s to 2.53s in the controlled local
comparison. Most importantly, the regenerated canonical artifact was byte
identical; the JSON checkpoint differed at roughly one floating-point ULP in
normalized strategy probabilities. New cloud stages therefore write
`.checkpoint.msgpack.gz`, while parent-stage discovery accepts either format.

A separate 50,000-iteration full-postflop export produced 961,899 served rows,
a 715.8MB canonical JSON stream, and a 110.9MB gzip artifact (115 compressed
bytes per exported row). Peak RSS was 2.42GB because training nodes and export
rows coexist while serializing. A linear 400,000-iteration storage projection
is therefore roughly 0.9GB compressed per seed, but promotion still uses the
actual binary/quantized size audit rather than this JSON projection.

The runner accepts `--hs-dcfr-30-horizon N`. This schedule is part of the
immutable training fingerprint and must remain identical across every resume
stage; `N` must be at least the final requested iteration count and no more
than 10M, preventing an accidental unbounded weight-vector allocation. On the final
300k midpoint screen with 128 root-deviation samples per hand class, a horizon
of 600k improved both seeds over fixed DCFR by `0.111230bb` and `0.127109bb`.
Pair mean improved 20.6%, cross-seed spread fell 89.3%, and action-EV precision
improved on both seeds; unknown continuation coverage was only 0.3–0.6
percentage points worse. At the exact 400k first-stage budget, HS beats the
former fixed-cloud profile (gamma 2, delay 40k) on both seeds by `0.027390bb`
and `0.040095bb`; pair mean is 6.96% lower and spread is 71.0% lower. Coverage
and action-EV precision tradeoffs are small and mixed. The exact 20bb command
therefore pins `N=600000` from its first stage and uses zero averaging delay,
matching the best tested pilots. The runner's generic default remains fixed
DCFR so other depths or experiments do not inherit the 20bb selection
accidentally. This records the best tested lineage; it does not override the
launch hold above.

## Build and preflight (hold-safe)

On a clean 64-bit Linux host:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config python3 git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
git clone https://github.com/davidvayn/pokersolver.git pokersolver
cd pokersolver/preflop-solver
cargo test --release
cargo build --release
cd ..
```

The required 800-round pair was run sequentially on the local host. Fixed DCFR
is intentional: after correcting preservation of tiny positive HS average
weights, a matched HS-DCFR pair still worsened root quality and cross-seed
stability. The measured table was roughly 12M sets per seed, so the two workers
were kept sequential on the 16GB machine. Reproduce its preflight with:

```bash
python3 preflop-solver/neural/cloud_blueprint_run.py \
  --binary "$PWD/preflop-solver/target/release/preflop-solver" \
  --output-dir "$PWD/preflop-solver/neural/runs/public-chance-20bb-i800" \
  --public-chance-sampling \
  --depth 20 --iterations 800 --seeds 26001,26002 \
  --max-concurrent 1 --max-information-sets 15000000 \
  --averaging-delay 0 --checkpoint-every 400 \
  --held-out-deals 2000 --root-deviation-samples 128 \
  --action-value-deals 2000 --dcfr-alpha 1.5 --dcfr-beta 0 \
  --dcfr-gamma 2 --no-export-postflop-strategies \
  --bytes-per-information-set 700 \
  --minimum-free-disk-gb 40 --dry-run
```

The completed direct pilot used the same solver configuration without durable
checkpoint writes. Remove `--dry-run` only when intentionally reproducing it
through the cloud runner.
The 700-byte pilot estimate is specific to the measured public-chance path
(roughly 445 bytes per live set at the larger 400-round seed) and retains
about 57% per-set headroom before the runner's additional 20% multiplier. Do
not reuse it for the scalar complete-export commands below, which retain the
conservative 2,300-byte default.
This is a preflop-export policy screen, not a serving artifact. Both seeds
improved root local deviation plus MAE, TV, primary agreement, and authentic
continuation coverage, so the next authorized paid experiment is the capped
1,600-round stage below. It starts a durable lineage from zero and still omits
the expensive complete postflop export:

```bash
python3 preflop-solver/neural/cloud_blueprint_run.py \
  --binary "$PWD/preflop-solver/target/release/preflop-solver" \
  --output-dir "$PWD/preflop-solver/neural/runs/public-chance-terminal-v03-20bb-i1600" \
  --public-chance-sampling --integrate-terminal-actions \
  --depth 20 --iterations 1600 --seeds 26001,26002 \
  --max-concurrent 2 --max-information-sets 45000000 \
  --averaging-delay 0 --checkpoint-every 800 \
  --held-out-deals 5000 --root-deviation-samples 256 \
  --action-value-deals 5000 --dcfr-alpha 1.5 --dcfr-beta 0 \
  --dcfr-gamma 2 --no-export-postflop-strategies \
  --bytes-per-information-set 700 \
  --minimum-free-disk-gb 100 --dry-run
```

Use a 128GB host for two concurrent workers. The 45M cap at 700 bytes per set,
two workers, and 20% extra headroom projects **75.6GB (70.4GiB)**. This is a
conservative preflop-export pilot estimate, not measured full-export/checkpoint
peak memory. Integration grows more states per round; reusing the old 25M cap
could stop the experiment early. Keep the generic 2,300-byte default for
complete-export jobs until the new path is measured at scale.

Consider 3,200 rounds only if the 1,600 pair improves both root values and at
least two stability or coverage families, the observed memory slope permits a
new cap, and paid time remains. Do not automatically extend a 45M-cap run or
assume both workers still fit. Before a long continuation, test a legal
full-hand responder against this exact frozen tabular policy; the existing
neural responder's old scores are not this model's exploitability. Complete
postflop export, serving-size audit, independent response testing, and
action-value precision remain distinct work. Nothing here promotes a model.

Dry-run the exact paired command and resource guard:

```bash
python3 preflop-solver/neural/cloud_blueprint_run.py \
  --binary "$PWD/preflop-solver/target/release/preflop-solver" \
  --output-dir "$PWD/preflop-solver/neural/runs/schema-v3-20bb-cloud" \
  --depth 20 --iterations 400000 --seeds 26001,26002 \
  --max-concurrent 2 --max-information-sets 15000000 \
  --averaging-delay 0 --checkpoint-every 100000 \
  --held-out-deals 10000 --root-deviation-samples 128 \
  --action-value-deals 10000 --dcfr-alpha 1.5 --dcfr-beta 0 \
  --dcfr-gamma 2 --hs-dcfr-30-horizon 600000 \
  --export-postflop-strategies \
  --minimum-free-disk-gb 100 --dry-run
```

The 400,000-iteration first stage is sized to approach but not deliberately
hit a 15-million-set guard if the measured growth curve continues. Do not turn
the original one-million-iteration sketch into a 15-million-set job: it would
predictably stop early. Inspect both first-stage summaries before increasing
the target and cap. If growth remains near the local curve, run larger-cap
seeds sequentially on 128GB or move the concurrent pair to a larger memory
instance.

The final local schedule pair completed this exact 400k midpoint at 11.17M and
11.08M information sets. Both seeds improved their matched policy diagnostic
from 300k, but the local pilot deliberately omitted the huge postflop export
and resumable checkpoint writes. A cloud stage would still be required to
produce the complete frozen profile and durable lineage artifacts, but the
known terminal policy failures now take precedence over that missing output.

An extension uses a new output directory and explicitly pins the completed
parent stage. Keep the binary, averaging delay, DCFR tuple, seeds, and action
grid unchanged. For example, if the observed first-stage growth justifies a
22-million-set cap, extend to 600,000 iterations sequentially with:

```bash
python3 preflop-solver/neural/cloud_blueprint_run.py \
  --binary "$PWD/preflop-solver/target/release/preflop-solver" \
  --resume-from-dir "$PWD/preflop-solver/neural/runs/schema-v3-20bb-cloud" \
  --output-dir "$PWD/preflop-solver/neural/runs/schema-v3-20bb-cloud-i600000" \
  --depth 20 --iterations 600000 --seeds 26001,26002 \
  --max-concurrent 1 --max-information-sets 22000000 \
  --averaging-delay 0 --checkpoint-every 100000 \
  --held-out-deals 10000 --root-deviation-samples 256 \
  --action-value-deals 10000 --dcfr-alpha 1.5 --dcfr-beta 0 \
  --dcfr-gamma 2 --hs-dcfr-30-horizon 600000 \
  --export-postflop-strategies \
  --minimum-free-disk-gb 100 --dry-run
```

The extension preflight requires a completed parent manifest, the same solver
binary and immutable solver settings, the exact seed set, and unchanged
checkpoint sizes and SHA-256 hashes. A local checkpoint in the extension
directory takes precedence after an interruption, so retries never restart
from the older parent stage.

The 600,000 target is the immutable horizon and terminal iteration of this
lineage, not an unconditional promotion point. Activate nothing unless both
seeds pass independent evaluation, stability, coverage, probability, and
storage gates. A checkpoint from this lineage cannot extend beyond 600k
without changing the schedule it has already executed. If evidence justifies
a longer run, first screen and pin a longer horizon, then start a fresh pair
from iteration zero in a new lineage; never rescale the schedule inside an
existing checkpoint.

The completed local terminal pair establishes the current hold. It passes
aggregate action delta (`1.0489pp <= 3pp`), median TV (`19.11% <= 20%`),
primary-action agreement (`85.21% >= 85%`), and the sample/visit floors. It
fails per-action MAE (`9.2088% > 5%`), p95 TV (`45.91% > 35%`), root local
deviation (`0.331/0.320bb > 0.10bb`), and continuation coverage. A screened
1M horizon is not the answer: at the same 600k midpoint it worsens MAE to
`10.2487%`, p95 TV to `49.54%`, aggregate delta to `2.2762%`, and max TV to
`65.29%`. It is rejected, so there is no unchecked 1M command in this runbook.

The first stage retains 128 root-deviation samples per class because it is a
midpoint continuation decision. The terminal 600k command raises this to 256,
matching the minimum local-deviation audit gate. Evaluation controls are not
part of checkpoint training state, so this change does not alter or invalidate
the frozen 400k regret/average tables. If the 10,000-deal action-value pass
still misses its precision target, preserve the trained checkpoint and run a
larger independent evaluation pass; do not restart training merely to increase
evaluation samples.

The runner supports that same-iteration pass explicitly. Point
`--resume-from-dir` at the completed training stage, use a new output directory,
keep every immutable training option identical, set `--iterations` to the
parent's completed count, add `--evaluation-only`, and increase only the
evaluation/export controls. For example:

```bash
python3 preflop-solver/neural/cloud_blueprint_run.py \
  --binary "$PWD/preflop-solver/target/release/preflop-solver" \
  --resume-from-dir "$PWD/preflop-solver/neural/runs/schema-v3-20bb-cloud-i600000" \
  --evaluation-only \
  --output-dir "$PWD/preflop-solver/neural/runs/schema-v3-20bb-cloud-i600000-eval50000" \
  --depth 20 --iterations 600000 --seeds 26001,26002 \
  --max-concurrent 1 --max-information-sets 22000000 \
  --averaging-delay 0 --checkpoint-every 100000 \
  --held-out-deals 50000 --root-deviation-samples 256 \
  --action-value-deals 50000 --dcfr-alpha 1.5 --dcfr-beta 0 \
  --dcfr-gamma 2 --hs-dcfr-30-horizon 600000 \
  --export-postflop-strategies --minimum-free-disk-gb 100
```

Rust does not alter the completed tables in this mode. The evaluation manifest
records and integrity-checks the immutable parent checkpoint rather than
copying or rewriting its multi-gigabyte contents. A paired same-iteration
integration smoke raised all three evaluation counts, switched from a
preflop-only diagnostic export to the complete trained export, and verified
that exported and trained information-set counts remained equal.

Do not pass `--allow-resource-oversubscription` merely to force the run onto a
smaller host. Lower `--max-concurrent` to 1 or reduce the information-set cap
instead. Two sequential seeds remain scientifically valid, though they take
twice as long.

## Launch, monitor, and resume (after the hold is lifted)

After recording why the hold was lifted, remove only `--dry-run`, then start
under the host's normal durable process manager (`systemd`, `tmux`, or an
equivalent). The runner writes:

- one immutable `.artifact.json.gz` per seed;
- one resumable `.checkpoint.json.gz` per seed (internal checkpoint schema 4;
  the trained policy/model remains schema v3);
- one compact `.summary.json` per seed with completion, table-size, held-out
  EV/error, root local-deviation/error, action-EV precision, coverage, and
  validation fields, plus the 169 root action distributions and root-only
  visit/update diagnostics, the executed averaging delay, DCFR tuple,
  schedule, and immutable schedule horizon;
- one combined stdout/stderr log per seed;
- an atomically replaced `run-manifest.json` with commands, PIDs, status,
  host/resource facts, exact binary SHA-256, a stable run fingerprint,
  explicit training-versus-evaluation-only provenance,
  compressed/decompressed sizes, and the SHA-256 of canonical decompressed
  JSON; after every seed passes, it also contains aggregate pair mean/spread,
  worst held-out and continuation coverage, minimum action-EV precision, and maximum table
  size. The aggregate also computes root-policy stability without loading the
  full artifacts: maximum aggregate action delta (`<=3pp`), combo-weighted
  per-action MAE (`<=5pp`), primary-action agreement (`>=85%`), complete
  169-class coverage, compatible actions, and probability-sum validity.

Each durable checkpoint also appends a concise progress line containing the
completed iteration, live information-set count, and checkpoint path. Monitor
the job without parsing a checkpoint using, for example,
`tail -F <output-dir>/*.log`; inspect process/seed state atomically in
`<output-dir>/run-manifest.json`. Absence of a new line between the configured
100k checkpoints is expected and is not by itself a stalled-worker signal.

The orchestrator streams every artifact to the end, validates the canonical
JSON hash, binds its prefix identity (schema, artifact ID, model, solver version,
and both config hashes) to the small summary, then fails the seed if it stopped at the
information-set ceiling or completed fewer than the requested iterations. It
also marks the individual seed failed if Rust exits successfully without any
required artifact, checkpoint, or summary, or if the artifact/summary predates
the current child attempt. The checkpoint is deliberately exempt from that
freshness check because a same-iteration evaluation retry must leave it
immutable. A successful process exit or gzip write alone is therefore not
considered a successful run.

Before launch, the runner rejects seeds outside Rust's `u64` domain (including
the derived evaluation-seed offsets), invalid evaluation counts or checkpoint
cadence, iteration/table-size requests outside the cloud binary's 64-bit
domains, negative or non-finite DCFR exponents, HS-DCFR with non-default base
exponents, and impossible disk or memory requests.
After launch, uncertainty fields must be nonnegative and every coverage
fraction must be finite and inside `[0,1]`.

For a full export, the runner also requires exported information-set count to
equal trained information-set count, verifies that preflop plus postflop counts
equal the total, and bounds evaluated/exported counts. A process that exits
zero after omitting trained postflop rows is therefore marked failed. The
preflop-only diagnostic switch remains valid, but its exported count may not
exceed its preflop information-set count.

SIGINT or SIGTERM is forwarded to active workers. Each worker also writes a
final checkpoint when it exits normally. Re-run the identical command after a
host interruption; existing compatible checkpoints are supplied through
`--resume`. Any configuration mismatch is rejected by the Rust trainer.
Before invoking Rust, the orchestrator also rejects an existing output
directory whose binary or training-setting fingerprint differs. A resume
cannot silently replace the original manifest with a different executable. If
training already reached the requested iteration, a validation retry reads the
existing checkpoint without rewriting the entire table.

Complete postflop strategy export is enabled by default. The
`--no-export-postflop-strategies` switch is only for resource diagnostics; its
artifact cannot become a full-hand model. A 10,000-iteration checkpoint cadence
would repeatedly rewrite the entire growing table and is intentionally not the
default. Choose a cadence that limits the job to a handful of full checkpoint
writes while meeting the acceptable restart-loss window.

The checked-in smoke procedure ran two seeds concurrently, read both gzip
streams to EOF, then resumed both checkpoints and reproduced identical
canonical artifact hashes. A real run still must be rejected if it reaches the
information-set ceiling before the target iteration count or if its subsequent
independent validation gates fail.

The final binary/runner integration smoke also launched independent fixed and
HS pairs through the orchestrator. Both variants completed, and the runner
matched the executed DCFR tuple, schedule name, immutable horizon, target
iterations, and early-stop state from each Rust sidecar before accepting the
manifests. This validates wiring and provenance only; the two-iteration smoke
artifacts are not policy evidence. A later compact-root smoke and deterministic
checkpoint retry confirmed that incomplete one-class policies are accepted as
intact training outputs but correctly fail closed as unavailable for
cross-seed stability. A 10,000-iteration integration pair reached all 169
classes and produced the same aggregate delta, per-action MAE, and primary
agreement as the standalone full-artifact comparator, while correctly failing
the two immature-policy gates.

The runner's isolated orchestration checks can be repeated without neural
dependencies:

```bash
cd preflop-solver/neural
python3 -m unittest discover -s . -p 'test_cloud_blueprint_run.py'
```

For preflop-only diagnostic artifacts, the standalone root comparator accepts
gzip directly, so no decompressed file copy is required:

```bash
npm run preflop:compare-blueprints -- \
  <output-dir>/hu-20bb-schema-v3-seed26001.artifact.json.gz \
  <output-dir>/hu-20bb-schema-v3-seed26002.artifact.json.gz
```

Add `--require-pass` so a failed root gate produces a nonzero exit status. The
standalone comparator materializes the JSON in memory; do not point it at a
multi-gigabyte full postflop export. The runner's compact sidecar aggregation
is the bounded-memory root gate for those cloud artifacts. Either root
comparison complements, but does not replace, the matched-reach postflop
stability and independent policy-quality audits.

## Promotion boundary

Training completion is not model activation. For each seed pair:

1. inspect the artifact's `stoppedEarly`, coverage, probability, and validation
   fields; use the compact sidecars first to compare held-out and root
   local-deviation metrics without loading the large policies;
2. run the independent exploitability/action-EV evaluation corpus;
3. compare cross-seed action frequencies only on matched reached states;
4. select the lower independently measured exploitability upper bound only if
   both seeds meet the stability and coverage gates;
5. export only the frozen average policy, then rerun quantization and hosted
   storage audits.

Keep 20bb isolated from future 50bb/100bb jobs. A depth can fail without
changing or delaying an already validated depth.
