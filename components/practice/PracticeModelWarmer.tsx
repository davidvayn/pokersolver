'use client';

import { useEffect } from 'react';
import { warmPracticeModels } from '@/lib/practice-models';

export function PracticeModelWarmer() {
  useEffect(() => {
    warmPracticeModels();
  }, []);

  return null;
}
