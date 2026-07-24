# Poker Lab Design System

This file records the design decisions selected after using the
`ui-ux-pro-max` recommendation workflow. It overrides the raw generated
suggestions where they conflicted with the established application.

## Product posture

Poker Lab is a serious, repeat-use training tool. Product screens should be
quiet, data-dense, and optimized for scanning and repeated decisions. Landing
pages may be more editorial, but must show real poker subject matter or a
faithful application preview.

## Color

Use the semantic CSS variables in `app/globals.css` through the matching
Tailwind tokens:

- Surfaces: `bg`, `surface`, `surface-2`, `border`
- Text: `fg`, `muted`
- Brand and focus: `accent`, `accent-fg`, `felt`
- Poker actions: `raise`, `call`, `fold`, `check`, `allin`
- Action foregrounds: `raise-fg`, `call-fg`, `fold-fg`, `allin-fg`

Both light and dark themes are first-class. Do not add page-local palettes or
assume that white text is accessible over every action color.

## Typography

Use the system sans stack for interface copy and the existing system monospace
stack for hand classes, percentages, measurements, and compact metadata. Keep
letter spacing at the default value. Reserve large type for true page heroes.

## Layout

- Use an 8px spacing rhythm with responsive 16px to 32px page gutters.
- Keep cards and panels at 8px radius or less.
- Do not nest cards inside cards.
- Keep operational pages unframed at the section level; cards are for repeated
  records, bounded tools, and modal content.
- Reserve space for the mobile bottom navigation.

## Interaction

- Use Lucide icons. Playing-card suit glyphs are allowed only as poker data.
- Keep interactive targets at least 44px high and provide visible focus styles.
- Use semantic buttons, links, labels, progress bars, and dialog roles.
- Do not make display-only matrices or charts keyboard-focusable.
- Use 150ms to 300ms color, opacity, or shadow transitions. Avoid
  layout-shifting hover effects.
- Respect reduced-motion preferences from `app/globals.css`.

## Data integrity

Poker training previews must use bundled reference data from
`data/preflop/ranges.ts` or be explicitly labeled as examples. The bundled
charts are simplified, curated 100bb baselines; do not present them as
authoritative solver output. Never present invented strategy frequencies or
personalized-looking performance data as the user's history.

## Landing page

The selected Deck direction uses a full-width playing-card scene with the
`Poker Lab` H1 and Practice as the primary action. Its surfaces follow the
active theme directly: light in light mode and dark in dark mode.

Use motion to focus attention, not as ambient noise. The two card-deal entrance
animations are brief and preserve a fully static experience under
`prefers-reduced-motion`.

## Validation

For every browser-facing change:

1. Run the affected flow with Browser Use.
2. Inspect 375px and 1440px viewports.
3. Check light and dark themes.
4. Confirm no horizontal overflow, fixed-element collisions, broken images,
   inert focus targets, or inaccessible names.
5. Run `npm test` and `npm run build`.
