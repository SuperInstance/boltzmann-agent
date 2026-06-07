//! Simulated annealing for energy minimization.
//!
//! Iterative optimization inspired by the annealing process in metallurgy.
//! Accepts worse solutions with Boltzmann probability, controlled by a
/// decreasing temperature schedule.
use serde::{Deserialize, Serialize};

use crate::rng::Xorshift64;
use crate::selection::TemperatureSchedule;

/// Result of a simulated annealing run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnealingResult {
    /// Best state found.
    pub best_state: Vec<f64>,
    /// Energy of the best state.
    pub best_energy: f64,
    /// Final state (may differ from best if last move was uphill).
    pub final_state: Vec<f64>,
    /// Energy of the final state.
    pub final_energy: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// History of best energies (sampled every `log_interval` steps).
    pub energy_history: Vec<f64>,
}

/// Configuration for simulated annealing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnealingConfig {
    /// Temperature schedule.
    pub schedule: TemperatureSchedule,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance: stop if best energy changes by less than this.
    pub tolerance: f64,
    /// Number of steps with no improvement before declaring convergence.
    pub patience: usize,
    /// Standard deviation of the Gaussian perturbation proposal.
    pub proposal_std: f64,
    /// Interval at which to log energy to history.
    pub log_interval: usize,
}

impl Default for AnnealingConfig {
    fn default() -> Self {
        Self {
            schedule: TemperatureSchedule::linear(10.0, 0.01, 10_000),
            max_iterations: 10_000,
            tolerance: 1e-8,
            patience: 500,
            proposal_std: 0.5,
            log_interval: 100,
        }
    }
}

/// Simulated annealing optimizer.
///
/// Minimizes an energy function `f(&[f64]) -> f64` over a state vector.
/// Uses iterative (not recursive) annealing with convergence tracking.
pub struct SimulatedAnnealing {
    config: AnnealingConfig,
    rng: Xorshift64,
}

impl SimulatedAnnealing {
    /// Create a new annealer with the given configuration.
    pub fn new(config: AnnealingConfig) -> Self {
        Self {
            config,
            rng: Xorshift64::default_seed(),
        }
    }

    /// Set the PRNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Xorshift64::new(seed);
        self
    }

    /// Run simulated annealing to minimize `energy_fn`, starting from `initial_state`.
    ///
    /// This is an iterative loop (NOT recursive). Each iteration:
    /// 1. Proposes a new state by perturbing the current state
    /// 2. Computes energy of the proposed state
    /// 3. Accepts if lower energy, or with Boltzmann probability if higher
    /// 4. Updates temperature according to schedule
    /// 5. Checks convergence
    pub fn minimize<F>(&mut self, energy_fn: F, initial_state: Vec<f64>) -> AnnealingResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut current_state = initial_state;
        let mut current_energy = energy_fn(&current_state);

        let mut best_state = current_state.clone();
        let mut best_energy = current_energy;

        let mut energy_history = Vec::new();
        let mut steps_without_improvement = 0;
        let mut converged = false;

        for iteration in 0..self.config.max_iterations {
            let temperature = self.config.schedule.temperature(iteration);

            // Propose new state: perturb one randomly chosen dimension
            let dim = self.rng.next_usize(current_state.len());
            let mut proposed_state = current_state.clone();
            proposed_state[dim] += self.rng.next_normal() * self.config.proposal_std;

            let proposed_energy = energy_fn(&proposed_state);

            // Metropolis acceptance criterion
            let delta_e = proposed_energy - current_energy;
            let accept = if delta_e <= 0.0 {
                true
            } else {
                let acceptance_prob = (-delta_e / temperature).exp();
                self.rng.next_f64() < acceptance_prob
            };

            if accept {
                current_state = proposed_state;
                current_energy = proposed_energy;

                if current_energy < best_energy {
                    best_state = current_state.clone();
                    let improvement = best_energy - current_energy;
                    best_energy = current_energy;
                    if improvement.abs() < self.config.tolerance {
                        steps_without_improvement += 1;
                    } else {
                        steps_without_improvement = 0;
                    }
                } else {
                    steps_without_improvement += 1;
                }
            } else {
                steps_without_improvement += 1;
            }

            // Log energy history
            if iteration % self.config.log_interval == 0 {
                energy_history.push(best_energy);
            }

            // Convergence check
            if steps_without_improvement >= self.config.patience {
                converged = true;
                break;
            }
        }

        energy_history.push(best_energy);

        AnnealingResult {
            best_state,
            best_energy,
            final_state: current_state,
            final_energy: current_energy,
            iterations: self.config.max_iterations,
            converged,
            energy_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple quadratic: f(x) = sum(x_i^2), minimum at origin with E=0.
    fn quadratic(state: &[f64]) -> f64 {
        state.iter().map(|x| x * x).sum()
    }

    /// Rastrigin-like function: many local minima, global at origin.
    fn rastrigin(state: &[f64]) -> f64 {
        state
            .iter()
            .map(|x| x * x - 10.0 * (2.0 * std::f64::consts::PI * x).cos() + 10.0)
            .sum()
    }

    /// Shifted quadratic: minimum at [3, 3].
    fn shifted_quadratic(state: &[f64]) -> f64 {
        state.iter().map(|x| (x - 3.0).powi(2)).sum()
    }

    #[test]
    fn test_finds_quadratic_minimum() {
        let config = AnnealingConfig {
            schedule: TemperatureSchedule::linear(5.0, 0.001, 50_000),
            max_iterations: 50_000,
            tolerance: 1e-12,
            patience: 5000,
            proposal_std: 0.3,
            log_interval: 1000,
        };
        let mut sa = SimulatedAnnealing::new(config).with_seed(42);
        let result = sa.minimize(quadratic, vec![5.0, 5.0]);

        assert!(
            result.best_energy < 0.1,
            "should find near-zero energy, got {}",
            result.best_energy
        );
        for x in &result.best_state {
            assert!(x.abs() < 1.0, "state should be near origin: {x}");
        }
    }

    #[test]
    fn test_finds_shifted_minimum() {
        let config = AnnealingConfig {
            schedule: TemperatureSchedule::linear(5.0, 0.001, 30_000),
            max_iterations: 30_000,
            tolerance: 1e-12,
            patience: 5000,
            proposal_std: 0.5,
            log_interval: 1000,
        };
        let mut sa = SimulatedAnnealing::new(config).with_seed(123);
        let result = sa.minimize(shifted_quadratic, vec![0.0, 0.0]);

        assert!(
            result.best_energy < 0.1,
            "should find near-zero energy, got {}",
            result.best_energy
        );
        for x in &result.best_state {
            assert!((x - 3.0).abs() < 1.0, "state should be near [3,3]: {x}");
        }
    }

    #[test]
    fn test_energy_history_decreases() {
        let config = AnnealingConfig {
            schedule: TemperatureSchedule::linear(5.0, 0.01, 5000),
            max_iterations: 5000,
            log_interval: 500,
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config).with_seed(42);
        let result = sa.minimize(quadratic, vec![10.0]);

        // Energy history should generally decrease
        assert!(result.energy_history.len() >= 2);
        assert!(
            result.energy_history.last().unwrap() <= &result.energy_history[0],
            "final energy should be <= initial"
        );
    }

    #[test]
    fn test_best_energy_leq_initial() {
        let config = AnnealingConfig {
            max_iterations: 1000,
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config).with_seed(99);
        let initial = vec![3.0, -2.0, 1.0];
        let initial_energy = quadratic(&initial);
        let result = sa.minimize(quadratic, initial);
        assert!(
            result.best_energy <= initial_energy,
            "best energy should not exceed initial"
        );
    }

    #[test]
    fn test_convergence_flag() {
        let config = AnnealingConfig {
            schedule: TemperatureSchedule::linear(1.0, 0.001, 100_000),
            max_iterations: 100_000,
            tolerance: 1e-8,
            patience: 10_000,
            proposal_std: 0.2,
            log_interval: 10000,
        };
        let mut sa = SimulatedAnnealing::new(config).with_seed(42);
        let result = sa.minimize(quadratic, vec![1.0]);
        // Should converge on simple quadratic
        assert!(result.converged, "should converge on simple problem");
    }

    #[test]
    fn test_rastrigin_does_not_crash() {
        let config = AnnealingConfig {
            schedule: TemperatureSchedule::linear(10.0, 0.01, 5000),
            max_iterations: 5000,
            proposal_std: 0.5,
            ..Default::default()
        };
        let mut sa = SimulatedAnnealing::new(config).with_seed(77);
        let result = sa.minimize(rastrigin, vec![5.0, 5.0]);
        assert!(result.best_energy.is_finite());
        assert_eq!(result.best_state.len(), 2);
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let config = AnnealingConfig {
            max_iterations: 1000,
            ..Default::default()
        };
        let mut sa1 = SimulatedAnnealing::new(config.clone()).with_seed(42);
        let mut sa2 = SimulatedAnnealing::new(config).with_seed(42);
        let r1 = sa1.minimize(quadratic, vec![2.0]);
        let r2 = sa2.minimize(quadratic, vec![2.0]);
        assert_eq!(r1.best_state, r2.best_state);
        assert!((r1.best_energy - r2.best_energy).abs() < 1e-15);
    }
}
