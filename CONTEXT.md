# Heads-up poker study

Solver and Practice study the same heads-up cash game. The difference is whether
the user studies public ranges or plays a dealt private hand.

## Language

**Hand engine**:
The rules of a complete hand: legal wagers, chip accounting, street transitions,
folds, all-ins, and showdown. It is distinct from the strategy used to choose actions.

**Policy**:
A distribution over modeled actions for a specific information set and game.
Supporting a stack depth in the rules does not establish trained policy coverage.

**Public range**:
The distribution over exact private-card combinations implied by public cards
and the preceding action trajectory, without revealing an opponent's dealt hand.

**Study line**:
A sequence of public actions across streets, with board cards selected by the
user. At an unresolved showdown, range study does not invent a private winner.
