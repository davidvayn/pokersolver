//! Exact cash checkdown geometry reuse; never a cache of ranges or values.
//! Ordered boards retain the existing river/combo/addition order. Keeping 64
//! boards covers all 49 turn proposals of a flop. Evicted/in-flight Arc owners
//! may live longer, but the cache itself retains at most 64 immutable kernels.
use super::*;

const MAX_BOARDS: usize = 64;
type KernelCell = Arc<OnceLock<Arc<CashCheckdownKernel>>>;

struct GeometryCache {
    clock: u64,
    capacity: usize,
    entries: BTreeMap<[u8; 4], (u64, KernelCell)>,
}

impl GeometryCache {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            clock: 0,
            capacity,
            entries: BTreeMap::new(),
        }
    }

    fn cell(&mut self, board: [u8; 4]) -> KernelCell {
        self.clock = self.clock.saturating_add(1);
        if let Some((used, cell)) = self.entries.get_mut(&board) {
            *used = self.clock;
            return cell.clone();
        }
        if self.entries.len() == self.capacity {
            let oldest = *self
                .entries
                .iter()
                .min_by_key(|(_, (used, _))| *used)
                .expect("nonempty geometry cache")
                .0;
            self.entries.remove(&oldest);
        }
        let cell = Arc::new(OnceLock::new());
        self.entries.insert(board, (self.clock, cell.clone()));
        cell
    }
}

pub(super) struct CashCheckdownKernel {
    // Reuse the exact existing rank/blocked-card/layout construction. This
    // solver is never trained or exposed. Its dummy game has no cash rules,
    // no queried beliefs/commitments, no model weights and no private cards.
    // Only immutable geometry methods are used; settlement accepts rules.
    geometry: TurnRiverSolver,
}

impl CashCheckdownKernel {
    fn new(board: [u8; 4]) -> Self {
        #[cfg(test)]
        cash_value::note_geometry_preparation();
        let config = TurnRiverSolveConfig {
            game: BlueprintConfig::default(),
            state: PublicBeliefState::turn_start(
                board,
                0,
                [1.0; 2],
                // Uniform support retains every board-legal counterfactual
                // query, even when actual own behavioral reach is zero.
                std::array::from_fn(|_| uniform_range(&board)),
            ),
            iterations: 2,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        };
        Self {
            geometry: TurnRiverSolver::new(config).expect("validated turn geometry"),
        }
    }

    pub(super) fn values(
        &self,
        rules: &crate::cash_game::CashGameRules,
        invested: [f64; 2],
        ranges: &[Vec<f64>; 2],
    ) -> [Vec<f64>; 2] {
        let cfvs = self
            .geometry
            .cash_showdown_values(rules, invested, ranges, None);
        std::array::from_fn(|player| {
            let masses =
                compatible_masses_from_card_marginals(&self.geometry.combos, &ranges[1 - player]);
            cfvs[player]
                .iter()
                .zip(masses)
                .map(|(value, mass)| if mass > 0.0 { value / mass } else { 0.0 })
                .collect()
        })
    }
}

pub(super) fn kernel(board: [u8; 4]) -> Arc<CashCheckdownKernel> {
    static CACHE: LazyLock<Mutex<GeometryCache>> =
        LazyLock::new(|| Mutex::new(GeometryCache::new(MAX_BOARDS)));
    let cell = CACHE.lock().expect("cash geometry cache").cell(board);
    // No global lock during expensive preparation. Concurrent requests for
    // a retained board share exactly one initialization via OnceLock.
    cell.get_or_init(|| Arc::new(CashCheckdownKernel::new(board)))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_cache_is_bounded_lru_and_preserves_in_flight_owners() {
        let mut cache = GeometryCache::new(2);
        let a = [1, 2, 3, 4];
        let b = [1, 2, 3, 5];
        let c = [1, 2, 3, 6];
        let first = cache.cell(a);
        let evicted = cache.cell(b);
        assert!(Arc::ptr_eq(&first, &cache.cell(a)));
        cache.cell(c);
        assert_eq!(cache.entries.len(), 2);
        assert!(!cache.entries.contains_key(&b));
        let new_b = cache.cell(b);
        assert!(!Arc::ptr_eq(&evicted, &new_b));
        assert_eq!(Arc::strong_count(&evicted), 1);
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn geometry_cold_concurrent_requests_share_one_kernel() {
        let cell = GeometryCache::new(1).cell([1, 6, 11, 16]);
        let builds = std::sync::atomic::AtomicUsize::new(0);
        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            let tasks = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        cell.get_or_init(|| {
                            builds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            Arc::new(CashCheckdownKernel::new([1, 6, 11, 16]))
                        })
                        .clone()
                    })
                })
                .collect::<Vec<_>>();
            let kernels = tasks
                .into_iter()
                .map(|task| task.join().unwrap())
                .collect::<Vec<_>>();
            assert!(kernels
                .iter()
                .all(|kernel| Arc::ptr_eq(kernel, &kernels[0])));
        });
        assert_eq!(builds.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn ordered_boards_are_not_silently_canonicalized() {
        let a = kernel([8, 13, 22, 31]);
        assert!(Arc::ptr_eq(&a, &kernel([8, 13, 22, 31])));
        assert!(!Arc::ptr_eq(&a, &kernel([13, 8, 22, 31])));
        assert!(!Arc::ptr_eq(&a, &kernel([8, 13, 22, 32])));
        assert!(a.geometry.nodes.is_empty());
        assert!(a.geometry.config.game.cash_rules.is_none());
    }

    #[test]
    fn cached_checkdown_preserves_fresh_solver_value_bits_and_house_ledger() {
        let cash_rules = crate::cash_game::study_rules("nl25").unwrap();
        let no_rake = crate::cash_game::study_rules("nl25-rake-off-control").unwrap();
        let combos = all_combos();
        for board in [[8, 13, 22, 31], [0, 4, 8, 12], [8, 9, 22, 23]] {
            let cached = kernel(board);
            for distribution in 0..3 {
                let mut ranges: [Vec<f64>; 2] = std::array::from_fn(|_| uniform_range(&board));
                if distribution == 1 {
                    for p in 0..2 {
                        for combo in &combos {
                            // Raw, unequal totals with many zero-own-reach
                            // queries; neither cached ranges nor EVs are valid.
                            ranges[p][combo.key()] *= if combo.key() % 5 == p {
                                0.0
                            } else {
                                ((combo.key() * 17 + p * 13) % 19 + 1) as f64 / 20.0
                            };
                        }
                    }
                } else if distribution == 2 {
                    ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
                    ranges[0][Combo::new(50, 51).key()] = 0.7;
                    ranges[1][Combo::new(44, 45).key()] = 0.2;
                    ranges[1][Combo::new(46, 47).key()] = 0.8;
                }
                for rules in [&cash_rules, &no_rake] {
                    for investment in [1.0, 7.52, 20.0] {
                        let invested = [investment; 2];
                        let mut game = BlueprintConfig::default();
                        game.small_blind_bb = 0.4;
                        game.effective_stack_bb = 20.0;
                        game.cash_rules = Some(rules.clone());
                        let fresh = TurnRiverSolver::new(TurnRiverSolveConfig {
                            game,
                            state: PublicBeliefState::turn_start(
                                board,
                                1,
                                invested,
                                std::array::from_fn(|_| uniform_range(&board)),
                            ),
                            iterations: 2,
                            averaging_delay: 0,
                            river_refinement_iterations: 0,
                            regret_matching_plus: false,
                        })
                        .unwrap();
                        let mut terminal = fresh.config.state.game_state();
                        terminal.terminal = Some(Terminal::Showdown);
                        let cfvs = fresh.cash_terminal_values(&terminal, &ranges, None);
                        let actual = cached.values(rules, invested, &ranges);
                        let masses: [Vec<f64>; 2] = std::array::from_fn(|p| {
                            compatible_masses_from_card_marginals(&combos, &ranges[1 - p])
                        });
                        for p in 0..2 {
                            for key in 0..COMBO_COUNT {
                                let expected = if masses[p][key] > 0.0 {
                                    cfvs[p][key] / masses[p][key]
                                } else {
                                    0.0
                                };
                                assert_eq!(actual[p][key].to_bits(), expected.to_bits());
                            }
                        }
                        let own: [f64; 2] = std::array::from_fn(|p| {
                            (0..COMBO_COUNT)
                                .map(|key| ranges[p][key] * masses[p][key] * actual[p][key])
                                .sum()
                        });
                        let gross = 2 * cash::cash_units(investment, rules).unwrap();
                        let house = rules.rake_units(gross, true).unwrap() as f64
                            / rules.units_per_bb as f64
                            * joint_compatibility_mass(&ranges);
                        assert!((own[0] + own[1] + house).abs() < 1e-11);
                        if distribution == 2 {
                            let absent = Combo::new(40, 41).key();
                            assert_eq!(ranges[0][absent], 0.0);
                            assert!(actual[0][absent].abs() > 0.001);
                        }
                    }
                }
            }
        }
    }
}
