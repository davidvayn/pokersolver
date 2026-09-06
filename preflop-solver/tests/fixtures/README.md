# Terminal flop development fixture

`terminal-flop-defender-b.msgpack` is a 159,594-byte MessagePack capture of
339 frozen average-policy rows, configuration, and a measured terminal
decision. It contains no regrets or resumable training state and is not a
serving model.

SHA-256: `4caf41a368b79a80ebdf576e792ed4cd5d1cc0203bde709428018904d0842dc9`.
Source checkpoint SHA-256:
`7a71a1b13975173af32aed3c610ae62a7c0fb25680ddf4397efef0bde220893d`.
The source is the 20bb, seed-26002, 800-round policy with the experimental
32-iteration flop / 64-iteration turn-river routing and 0.5 terminal blend.

The inspected delayed-attack holdout decision is source B, index 3, seat 0:
own cards `[34,21]`, flop `[27,2,9]`, preflop limp/check, then a 19bb BB shove.
The full checkpoint and compact replay produce identical action mixes,
posterior weights, sampled LBR values, and enumerated conditional values.
Enumerating 990 runouts for every compatible opponent holding yields call EV
-9.930968570274786bb, versus -1bb for folding. The old 75/25 fold/call mix
loses 2.2327421425686964bb conditionally. These are local values against the
frozen posterior, not full-game exploitability.

The adjacent Rust regression exercises the real policy composition path
under control and explicit full-weight correction settings. This is an
already-inspected development case, never untouched validation. The capture
command, source evidence, resource limits, and interpretation are recorded in
`docs/solver/overnight-2026-09-05.md` at the repository root.
