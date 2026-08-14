export interface RawActionFrequency {
  fold: number;
  shove: number;
}

export interface RawResponseFrequency {
  fold: number;
  call: number;
}

export interface RawExactComboStrategy {
  combo_key: number;
  cards: [number, number];
  card_names: [string, string];
  label: string;
  small_blind: RawActionFrequency;
  big_blind_vs_shove: RawResponseFrequency;
}

export interface RawHandClassStrategy {
  label: string;
  combo_count: number;
  small_blind: RawActionFrequency;
  big_blind_vs_shove: RawResponseFrequency;
}

export interface RawPushFoldArtifact {
  schema_version: number;
  artifact_id?: string;
  config_hash?: string;
  solver_version: string;
  model: string;
  generated_at_unix_seconds: number;
  payoff_convention: string;
  config: {
    small_blind_bb: number;
    big_blind_bb: number;
    effective_stack_bb: number;
    iterations: number;
    equity_samples: number;
    seed: number;
  };
  metrics: {
    profile_small_blind_ev_bb: number;
    small_blind_best_response_ev_bb: number;
    small_blind_ev_vs_big_blind_best_response_bb: number;
    nash_conv_bb: number;
    exploitability_bb: number;
    small_blind_best_response_equity_interval_bb: {
      low: number;
      high: number;
    };
    small_blind_ev_vs_big_blind_best_response_equity_interval_bb: {
      low: number;
      high: number;
    };
    nash_conv_equity_interval_bb: { low: number; high: number };
    equity_standard_error_upper_bound: number;
    called_payoff_standard_error_upper_bound_bb: number;
    compatible_deals: number;
    training_equity_cache_entries: number;
    evaluation_equity_cache_entries: number;
    evaluation_seed: number;
  };
  validation: {
    status: string;
    quality: string;
    validation_version: string;
    note: string;
    checks: Array<{
      name: string;
      passed: boolean;
      value: number;
      threshold: number;
      comparison: string;
    }>;
  };
  strategies: {
    exact_combos: RawExactComboStrategy[];
    hand_classes: RawHandClassStrategy[];
  };
}

export interface RawPushFoldActionValuesArtifact {
  schema_version: number;
  model: string;
  source_artifact_id: string;
  source_config_hash: string;
  source_artifact_sha256: string;
  evaluation_seed: number;
  equity_samples: number;
  called_payoff_standard_error_upper_bound_bb: number;
  hand_classes: Array<{
    label: string;
    combo_count: number;
    small_blind: {
      fold_ev_bb: number;
      shove_ev_bb: number;
      fold_standard_error_bb: number;
      shove_standard_error_upper_bound_bb: number;
    };
    big_blind_vs_shove: {
      fold_ev_bb: number;
      call_ev_bb: number;
      fold_standard_error_bb: number;
      call_standard_error_upper_bound_bb: number;
    };
  }>;
}

export interface CompactPushFoldScenario {
  artifact_id: string;
  config_hash: string;
  solver_version: string;
  model: string;
  generated_at_unix_seconds: number;
  effective_stack_bb: number;
  iterations: number;
  equity_samples: number;
  seed: number;
  exploitability_bb: number;
  quality: string;
  source_sha256: string;
  action_values_source_sha256: string;
  action_value_standard_error_upper_bound_bb: number;
  hands: Array<[label: string, shove: number, call: number]>;
  action_values: Array<[
    label: string,
    smallBlindFoldEvBb: number,
    smallBlindShoveEvBb: number,
    bigBlindFoldEvBb: number,
    bigBlindCallEvBb: number,
  ]>;
}
