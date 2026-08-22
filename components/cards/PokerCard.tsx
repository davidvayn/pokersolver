import { cardRank, cardSuit, RANKS, type Card } from '@/lib/cards';

export const CARD_SUIT_GLYPHS = ['♣', '♦', '♥', '♠'] as const;
export const CARD_SUIT_NAMES = ['clubs', 'diamonds', 'hearts', 'spades'] as const;

type PokerCardSize = 'practice' | 'sm' | 'md' | 'lg';

export function PokerCard({
  card,
  hidden = false,
  size = 'practice',
}: {
  card?: Card;
  hidden?: boolean;
  size?: PokerCardSize;
}) {
  const sizeClass = size === 'practice' ? '' : `playing-card-size-${size}`;

  if (hidden) {
    return (
      <span
        className={`playing-card playing-card-back ${sizeClass}`}
        aria-label="Face-down card"
      >
        <span className="playing-card-back-inner" aria-hidden="true">
          PL
        </span>
      </span>
    );
  }

  if (card === undefined) {
    return (
      <span
        className={`playing-card playing-card-empty ${sizeClass}`}
        aria-hidden="true"
      />
    );
  }

  const rank = RANKS[cardRank(card)];
  const suitIndex = cardSuit(card);
  const suit = CARD_SUIT_GLYPHS[suitIndex];

  return (
    <span
      className={`playing-card playing-card-suit-${suitIndex} ${sizeClass}`}
      aria-label={`${rank} of ${CARD_SUIT_NAMES[suitIndex]}`}
    >
      <span
        className="playing-card-corner playing-card-corner-top"
        aria-hidden="true"
      >
        <strong>{rank}</strong>
        <span>{suit}</span>
      </span>
      <span className="playing-card-center" aria-hidden="true">
        {suit}
      </span>
      <span
        className="playing-card-corner playing-card-corner-bottom"
        aria-hidden="true"
      >
        <strong>{rank}</strong>
        <span>{suit}</span>
      </span>
    </span>
  );
}
