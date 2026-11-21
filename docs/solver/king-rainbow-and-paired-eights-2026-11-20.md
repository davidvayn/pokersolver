# Broader king-rainbow and paired-eights flops — 2026-11-20

Two conditional, rake-free flop-to-river studies expand the saved library.
Each keeps exact cards and ordered runouts, half-pot and pot bets, one
modeled pot raise, and all-in. The declared weighted ranges are illustrative
inclusion priors, not preflop policy or action percentages. Existing
policies and the live full-hand resolver are unchanged.

| Study | Flop | Pot / behind | BTN / BB combos | Compatible deals | Ordered deal–runouts |
| --- | --- | ---: | ---: | ---: | ---: |
| [50bb king-high rainbow](../../data/solved-spots/flop-50bb-king-rainbow.json) | Kh 9d 5c | 32 / 34bb | 32 / 31 | 859 | 1,700,820 |
| [100bb paired eights with clubs](../../data/solved-spots/flop-100bb-paired-eights-clubs.json) | 8c 8d 6c | 60 / 70bb | 32 / 32 | 872 | 1,726,560 |

The king-high flop mixes sets, top pairs, two pair, ace-high and gutshots.
The paired two-tone flop adds quads, full houses, trips and club draws in
the same ranges as overpairs and unpaired hands. Blocked combinations are
removed before normalizing; the joint deal excludes overlapping private cards.

## Acceptance and serving

Both allocation-free audits found 104,872 public decisions and 250,374
total nodes. Sparse regret/average arrays are 106,138,368 and 107,823,104
bytes, respectively, inside the unchanged 512MiB budget. Each game fits
the existing 32-combo per-seat and 500,000-decision caps.

The existing native exporter ran 256 updates per game and freshly
re-imported the actual published bytes. No tolerance or abstraction changed.

| Study | Summed best-response gain (bb/spot) | Max action-EV error (bb) | Binary bytes | Manifest bytes | Largest shard |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50bb king-high rainbow | 0.03378644499817 | 0.00000190734863 | 77,978,559 | 235,980 | 46,066 |
| 100bb paired eights with clubs | 0.03079296889636 | 0.00000381469728 | 79,664,355 | 236,041 | 46,650 |

Both pass the 0.05bb summed response-gain and 0.0001bb available-action EV
checks. The maximum probability sum error is 4.85e-8. The scores refer to
these exact starting ranges and modeled actions; neither is a certificate for
full-hand play or for every off-path decision.

- 50bb manifest: `e43184db6ecd7aa259111e0da880b8de16fdfa86d7afbe166d809fab6ad4b844`.
- 100bb manifest: `0d0104ace6c0a311892d81f4f5c2f9ef1bea5e8938a7f1c5c6c2573a65ff8b9e`.

The concurrent exports, including re-imports, took 562.58s and 563.10s wall
with peak physical footprints of about 681MB and 684MB. These are local
observations, not production timing or general compute estimates.

The library now has 33 studies: 25 flops, four turns and four rivers. There
are 67,039 serving files totaling 1,945,318,209 uncompressed bytes. Each new
game contains 1,957 binary shards under the unchanged 65,536-byte cap plus
one immutable manifest. The earlier catalog entries and artifacts are
unchanged. Hands fetch only the pinned manifest and needed shards.

## Verification

`vitest run lib/solved-flop.test.ts -t 'flop-(50bb-king-rainbow|100bb-paired-eights-clubs)'`
passed 12 checks, including independent exhaustive EVs and both best responses,
all legal branches, exact joint-prior dealing, bounded decoding, and retry.
The final `next build` passed production compilation, type checking, and
generation of all 14 static pages. The full CI suite was not run.

Isolated, muted headless Chrome completed both spots through showdown: the
50bb BB checked every street; the 100bb BTN called 30bb on the flop and
checked turn and river. A blocked 50bb binary shard paused play without
feedback. Retry kept the original cards and action, then graded it exactly
once. The two pinned manifests served immutable one-year cache headers and
COOP/COEP. No live resolver request or application exception was observed;
the existing unrelated favicon 404 remained.

The longer 100bb option exposed intrinsic-width clipping in the mobile spot
picker. The control now shrinks within its container without changing option
text or semantics. Bounds, 44px target height and visible focus were checked
at 375, 768, 1024 and 1440px plus 844×375 landscape, in light and dark.
Desktop used normal motion and mobile used reduced motion. The 48px mobile
action dock remained above navigation. The browser verified keyboard action
and review focus; native selection used a DOM change event because the headless
native dropdown keyboard path could not be verified.
