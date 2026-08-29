import { beforeEach, describe, expect, it } from 'vitest';
import { useUi } from '@/lib/ui-store';

describe('UI settings', () => {
  beforeEach(() => {
    useUi.setState({ settingsOpen: false, showSolverStats: false });
  });

  it('keeps solver diagnostics opt-in', () => {
    expect(useUi.getState().showSolverStats).toBe(false);

    useUi.getState().setShowSolverStats(true);

    expect(useUi.getState().showSolverStats).toBe(true);
  });
});
