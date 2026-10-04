'use client';

import {
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
} from 'react';
import { createPortal } from 'react-dom';
import { Check, ChevronDown } from 'lucide-react';

export interface SelectOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
}

interface SelectProps {
  label: string;
  value: string;
  options: readonly SelectOption[];
  onChange: (value: string) => void;
  disabled?: boolean;
  className?: string;
  hideLabel?: boolean;
}

/** A non-editable combobox. Focus stays on its trigger, including inside dialogs. */
export function Select({
  label,
  value,
  options,
  onChange,
  disabled,
  className = '',
  hideLabel = false,
}: SelectProps) {
  const id = useId();
  const trigger = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const typeAhead = useRef({ text: '', time: 0 });
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [position, setPosition] = useState<CSSProperties>({
    left: 0,
    top: 0,
    width: 0,
    maxHeight: 280,
  });
  const selected = options.findIndex((option) => option.value === value);
  const selectedOption = options[selected];
  const enabled = options.flatMap((option, index) =>
    option.disabled ? [] : [index]
  );

  function show(
    index = selected >= 0 && !options[selected]?.disabled
      ? selected
      : (enabled[0] ?? 0)
  ) {
    if (disabled || !enabled.length) return;
    place();
    setActive(index);
    setOpen(true);
  }

  function choose(index: number) {
    const option = options[index];
    if (!option || option.disabled) return;
    onChange(option.value);
    setOpen(false);
    trigger.current?.focus({ preventScroll: true });
  }

  function place() {
    const box = trigger.current?.getBoundingClientRect();
    if (!box) return;
    const below = window.innerHeight - box.bottom - 12;
    const above = box.top - 12;
    const down = below >= Math.min(280, above);
    const height = Math.max(44, Math.min(280, down ? below : above));
    const width = Math.min(Math.max(box.width, 200), window.innerWidth - 24);
    setPosition({
      left: Math.max(12, Math.min(box.left, window.innerWidth - width - 12)),
      ...(down
        ? { top: box.bottom + 6 }
        : { bottom: window.innerHeight - box.top + 6 }),
      width,
      maxHeight: height,
    });
  }

  useEffect(() => {
    if (!open) return;
    function outside(event: PointerEvent) {
      const target = event.target as Node;
      if (!trigger.current?.contains(target) && !menu.current?.contains(target))
        setOpen(false);
    }
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    document.addEventListener('pointerdown', outside);
    return () => {
      window.removeEventListener('resize', place);
      window.removeEventListener('scroll', place, true);
      document.removeEventListener('pointerdown', outside);
    };
  }, [open]);

  useEffect(() => {
    if (open)
      menu.current
        ?.querySelector<HTMLElement>(`[data-option-index="${active}"]`)
        ?.scrollIntoView({ block: 'nearest' });
  }, [active, open]);

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === 'Escape' && open) {
      event.preventDefault();
      event.stopPropagation();
      setOpen(false);
      return;
    }
    if (event.key === 'Tab') {
      setOpen(false);
      return;
    }
    if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      const direction = event.key === 'ArrowUp' ? -1 : 1;
      const current = enabled.indexOf(active);
      const next =
        event.key === 'Home'
          ? enabled[0]
          : event.key === 'End'
            ? enabled.at(-1)
            : enabled[(current + direction + enabled.length) % enabled.length];
      if (!open)
        show(
          event.key === 'Home'
            ? enabled[0]
            : event.key === 'End'
              ? enabled.at(-1)
              : undefined
        );
      else if (next !== undefined) setActive(next);
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (open) choose(active);
      else show();
      return;
    }
    if (
      event.key.length === 1 &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey
    ) {
      event.preventDefault();
      const now = Date.now();
      const previous =
        now - typeAhead.current.time < 700 ? typeAhead.current.text : '';
      const text = (previous + event.key).toLowerCase();
      typeAhead.current = { text, time: now };
      const start = open ? active : selected;
      const ordered = [
        ...enabled.filter((index) => index > start),
        ...enabled.filter((index) => index <= start),
      ];
      const match = ordered.find((index) =>
        options[index].label.toLowerCase().startsWith(text)
      );
      if (match !== undefined) show(match);
    }
  }

  return (
    <div className={`ui-select ${className}`}>
      <span
        id={`${id}-label`}
        className={hideLabel ? 'sr-only' : 'ui-field-label'}
      >
        {label}
      </span>
      <button
        ref={trigger}
        type="button"
        role="combobox"
        aria-labelledby={`${id}-label ${id}-value`}
        aria-controls={open ? `${id}-list` : undefined}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-activedescendant={open ? `${id}-option-${active}` : undefined}
        disabled={disabled || !enabled.length}
        onKeyDown={onKeyDown}
        onClick={() => (open ? setOpen(false) : show())}
        className="ui-select-trigger"
      >
        <span id={`${id}-value`} className="min-w-0 truncate">
          {selectedOption?.label ?? 'Choose'}
        </span>
        <ChevronDown
          aria-hidden="true"
          className={`h-4 w-4 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>
      {open &&
        typeof document !== 'undefined' &&
        createPortal(
          <div
            ref={menu}
            id={`${id}-list`}
            role="listbox"
            aria-labelledby={`${id}-label`}
            className="ui-select-menu"
            style={position}
          >
            {options.map((option, index) => (
              <div
                key={option.value}
                id={`${id}-option-${index}`}
                role="option"
                aria-selected={option.value === value}
                aria-disabled={option.disabled || undefined}
                data-option-index={index}
                data-active={active === index}
                onPointerMove={() => !option.disabled && setActive(index)}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => choose(index)}
                className="ui-select-option"
              >
                <span className="min-w-0 flex-1">
                  <span className="block break-words">{option.label}</span>
                  {option.description && (
                    <span className="mt-0.5 block text-sm text-muted">
                      {option.description}
                    </span>
                  )}
                </span>
                {option.value === value && (
                  <Check
                    className="h-4 w-4 shrink-0 text-accent"
                    aria-hidden="true"
                  />
                )}
              </div>
            ))}
          </div>,
          trigger.current?.closest('[role="dialog"]') ?? document.body
        )}
    </div>
  );
}
