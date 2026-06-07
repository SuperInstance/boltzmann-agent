//! Action selection via Boltzmann sampling with temperature annealing.
//!
//! Given a set of actions with associated energies (costs), select actions
//! by sampling from the Boltzmann distribution. Temperature controls the
//! exploration-exploitation tradeoff.

use serde::{Deserialize, Serialize};

use crate::distribution::BoltzmannDistribution;
use crate::rng::Xorshift64;

/// Temperature annealing schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemperatureSchedule {
    /// Linear annealing: T(t) = T_start - rate * t, clamped at T_min.
    Linear { t_start: f64, t_min: f64, rate: f64 },
    /// Exponential annealing: T(t) = T_start * decay^t.
    Exponential {
        t_start: f64,
        t_min: f64,
        decay: f64,
    },
    /// Cosine annealing: T(t) = T_min + 0.5*(T_start - T_min)*(1 + cos(π*t/t_end)).
    Cosine {
        t_start: f64,
        t_min: f64,
        t_end: f64,
    },
}

impl TemperatureSchedule {
    /// Compute temperature at step `step`.
    pub fn temperature(&self, step: usize) -> f64 {
        match self {
            Self::Linear {
                t_start,
                t_min,
                rate,
            } => {
                let t = t_start - rate * step as f64;
                t.max(*t_min)
            }
            Self::Exponential {
                t_start,
                t_min,
                decay,
            } => {
                let t = t_start * decay.powi(step as i32);
                t.max(*t_min)
            }
            Self::Cosine {
                t_start,
                t_min,
                t_end,
            } => {
                if *t_end <= 0.0 {
                    return *t_min;
                }
                let progress = (step as f64 / *t_end).min(1.0);
                let cosine = (std::f64::consts::PI * progress).cos();
                t_min + 0.5 * (t_start - t_min) * (1.0 + cosine)
            }
        }
    }

    /// Create a linear schedule from start to min over n steps.
    pub fn linear(t_start: f64, t_min: f64, n_steps: usize) -> Self {
        let rate = if n_steps > 0 {
            (t_start - t_min) / n_steps as f64
        } else {
            0.0
        };
        Self::Linear {
            t_start,
            t_min,
            rate,
        }
    }

    /// Create an exponential schedule with given half-life.
    pub fn exponential(t_start: f64, t_min: f64, half_life: f64) -> Self {
        let decay = 0.5_f64.powf(1.0 / half_life);
        Self::Exponential {
            t_start,
            t_min,
            decay,
        }
    }
}

/// Boltzmann action selector.
///
/// Selects actions by sampling from the Boltzmann distribution over action energies.
/// Low temperature → exploit (select lowest-energy actions). High temperature → explore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSelector {
    /// Temperature schedule.
    schedule: TemperatureSchedule,
    /// Current step counter.
    step: usize,
    /// PRNG for sampling.
    #[serde(skip)]
    rng: Option<Xorshift64>,
}

impl ActionSelector {
    /// Create a new action selector with the given temperature schedule.
    pub fn new(schedule: TemperatureSchedule) -> Self {
        Self {
            schedule,
            step: 0,
            rng: None,
        }
    }

    /// Set the PRNG seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Some(Xorshift64::new(seed));
        self
    }

    /// Get the current temperature.
    pub fn temperature(&self) -> f64 {
        self.schedule.temperature(self.step)
    }

    /// Get the current step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Select an action from the given energies using Boltzmann sampling.
    ///
    /// Returns the index of the selected action. Advances the step counter.
    pub fn select(&mut self, energies: &[f64]) -> usize {
        let temp = self.temperature();
        let dist = BoltzmannDistribution::new(energies.to_vec(), temp);
        let rng = self.rng.get_or_insert_with(Xorshift64::default_seed);
        let idx = dist.sample(rng);
        self.step += 1;
        idx
    }

    /// Select the greedy (lowest-energy) action deterministically.
    pub fn select_greedy(&self, energies: &[f64]) -> usize {
        energies
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Get Boltzmann probabilities for the current temperature.
    pub fn probabilities(&self, energies: &[f64]) -> Vec<f64> {
        let temp = self.temperature();
        let dist = BoltzmannDistribution::new(energies.to_vec(), temp);
        dist.probabilities().to_vec()
    }

    /// Reset to step 0.
    pub fn reset(&mut self) {
        self.step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_linear_schedule() {
        let schedule = TemperatureSchedule::linear(10.0, 0.0, 10);
        assert!((schedule.temperature(0) - 10.0).abs() < TOL);
        assert!((schedule.temperature(5) - 5.0).abs() < TOL);
        assert!((schedule.temperature(10) - 0.0).abs() < TOL);
        assert!((schedule.temperature(20) - 0.0).abs() < TOL); // clamped
    }

    #[test]
    fn test_exponential_schedule() {
        let schedule = TemperatureSchedule::exponential(10.0, 0.01, 10.0);
        // At step 10 (one half-life), T ≈ 5.0
        let t = schedule.temperature(10);
        assert!((t - 5.0).abs() < 0.1, "t at half-life = {t}");
    }

    #[test]
    fn test_cosine_schedule() {
        let schedule = TemperatureSchedule::Cosine {
            t_start: 10.0,
            t_min: 0.0,
            t_end: 100.0,
        };
        assert!((schedule.temperature(0) - 10.0).abs() < TOL);
        assert!((schedule.temperature(100) - 0.0).abs() < TOL);
        // Midpoint should be roughly halfway
        let mid = schedule.temperature(50);
        assert!((mid - 5.0).abs() < TOL, "cosine midpoint = {mid}");
    }

    #[test]
    fn test_select_produces_valid_index() {
        let schedule = TemperatureSchedule::Linear {
            t_start: 1.0,
            t_min: 0.1,
            rate: 0.01,
        };
        let mut selector = ActionSelector::new(schedule).with_seed(42);
        let energies = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for _ in 0..100 {
            let idx = selector.select(&energies);
            assert!(idx < 5, "index {idx} out of bounds");
        }
    }

    #[test]
    fn test_greedy_selects_lowest() {
        let schedule = TemperatureSchedule::linear(1.0, 0.1, 10);
        let selector = ActionSelector::new(schedule);
        let energies = vec![5.0, 1.0, 3.0, 0.5, 2.0];
        assert_eq!(selector.select_greedy(&energies), 3);
    }

    #[test]
    fn test_low_temp_mostly_greedy() {
        let schedule = TemperatureSchedule::Linear {
            t_start: 0.01,
            t_min: 0.01,
            rate: 0.0,
        };
        let mut selector = ActionSelector::new(schedule).with_seed(42);
        let energies = vec![0.0, 10.0, 20.0];
        let mut counts = vec![0usize; 3];
        for _ in 0..1000 {
            counts[selector.select(&energies)] += 1;
        }
        assert!(
            counts[0] > 900,
            "low temp should be mostly greedy: {counts:?}"
        );
    }

    #[test]
    fn test_high_temp_explores() {
        let schedule = TemperatureSchedule::Linear {
            t_start: 100.0,
            t_min: 100.0,
            rate: 0.0,
        };
        let mut selector = ActionSelector::new(schedule).with_seed(42);
        let energies = vec![0.0, 10.0, 20.0];
        let mut counts = vec![0usize; 3];
        for _ in 0..3000 {
            counts[selector.select(&energies)] += 1;
        }
        // At high T, all actions should be selected roughly equally
        assert!(counts[2] > 500, "high temp should explore: {counts:?}");
    }

    #[test]
    fn test_step_advances() {
        let schedule = TemperatureSchedule::linear(10.0, 1.0, 100);
        let mut selector = ActionSelector::new(schedule);
        assert_eq!(selector.step(), 0);
        selector.select(&[1.0, 2.0]);
        assert_eq!(selector.step(), 1);
        selector.select(&[1.0, 2.0]);
        assert_eq!(selector.step(), 2);
    }

    #[test]
    fn test_reset() {
        let schedule = TemperatureSchedule::linear(10.0, 1.0, 100);
        let mut selector = ActionSelector::new(schedule);
        for _ in 0..10 {
            selector.select(&[1.0, 2.0]);
        }
        assert_eq!(selector.step(), 10);
        selector.reset();
        assert_eq!(selector.step(), 0);
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let schedule = TemperatureSchedule::Linear {
            t_start: 2.0,
            t_min: 0.1,
            rate: 0.01,
        };
        let selector = ActionSelector::new(schedule);
        let probs = selector.probabilities(&[1.0, 2.0, 3.0]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }
}
