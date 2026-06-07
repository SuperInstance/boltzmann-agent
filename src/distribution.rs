//! Boltzmann distribution: P(s) = exp(-βE(s)) / Z
//!
//! Core module implementing the Boltzmann (Gibbs) distribution over discrete states
//! parameterized by energy values and a temperature parameter.

use serde::{Deserialize, Serialize};

use crate::rng::Xorshift64;

/// The Boltzmann distribution over discrete states.
///
/// Given energy values E_i for each state i and a temperature T,
/// the probability of each state is:
///
/// ```text
/// P(i) = exp(-E_i / (k·T)) / Z
/// ```
///
/// where Z = Σ_i exp(-E_i / (k·T)) is the partition function and k is Boltzmann's constant.
/// In this implementation we set k = 1 (natural units), so β = 1/T.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoltzmannDistribution {
    /// Energy of each state.
    energies: Vec<f64>,
    /// Temperature parameter T. Must be positive.
    temperature: f64,
    /// Boltzmann constant k. Defaults to 1.0 (natural units).
    boltzmann_constant: f64,
    /// Cached partition function Z.
    partition_function: f64,
    /// Cached probabilities.
    probabilities: Vec<f64>,
}

impl BoltzmannDistribution {
    /// Create a new Boltzmann distribution from energies and temperature.
    ///
    /// # Panics
    /// Panics if temperature <= 0 or energies is empty.
    pub fn new(energies: Vec<f64>, temperature: f64) -> Self {
        assert!(!energies.is_empty(), "energies must not be empty");
        assert!(temperature > 0.0, "temperature must be positive");
        Self::new_with_k(energies, temperature, 1.0)
    }

    /// Create with a custom Boltzmann constant k.
    pub fn new_with_k(energies: Vec<f64>, temperature: f64, boltzmann_constant: f64) -> Self {
        assert!(!energies.is_empty(), "energies must not be empty");
        assert!(temperature > 0.0, "temperature must be positive");
        assert!(
            boltzmann_constant > 0.0,
            "boltzmann constant must be positive"
        );

        let beta = 1.0 / (boltzmann_constant * temperature);
        let log_weights: Vec<f64> = energies.iter().map(|e| -beta * e).collect();

        // Numerically stable partition function via log-sum-exp
        let max_log = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let shift_sum: f64 = log_weights.iter().map(|w| (w - max_log).exp()).sum();
        let log_z = max_log + shift_sum.ln();
        let z = log_z.exp();

        let probabilities: Vec<f64> = log_weights.iter().map(|w| (w - log_z).exp()).collect();

        Self {
            energies,
            temperature,
            boltzmann_constant,
            partition_function: z,
            probabilities,
        }
    }

    /// Get the probabilities P(i) for each state.
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Get the partition function Z.
    pub fn partition_function(&self) -> f64 {
        self.partition_function
    }

    /// Get the inverse temperature β = 1/(kT).
    pub fn beta(&self) -> f64 {
        1.0 / (self.boltzmann_constant * self.temperature)
    }

    /// Compute the mean energy: ⟨E⟩ = Σ_i P(i)·E_i
    pub fn mean_energy(&self) -> f64 {
        self.probabilities
            .iter()
            .zip(self.energies.iter())
            .map(|(p, e)| p * e)
            .sum()
    }

    /// Compute the energy variance: Var(E) = ⟨E²⟩ - ⟨E⟩²
    pub fn variance(&self) -> f64 {
        let mean = self.mean_energy();
        let mean_sq: f64 = self
            .probabilities
            .iter()
            .zip(self.energies.iter())
            .map(|(p, e)| p * e * e)
            .sum();
        mean_sq - mean * mean
    }

    /// Compute the Shannon entropy: S = -Σ_i P(i)·ln(P(i))
    pub fn entropy(&self) -> f64 {
        self.probabilities
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| -p * p.ln())
            .sum()
    }

    /// Get the energy values.
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// Get the temperature.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Get the number of states.
    pub fn n_states(&self) -> usize {
        self.energies.len()
    }

    /// Sample a state index according to the Boltzmann probabilities.
    pub fn sample(&self, rng: &mut Xorshift64) -> usize {
        let u = rng.next_f64();
        let mut cumulative = 0.0;
        for (i, p) in self.probabilities.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                return i;
            }
        }
        self.probabilities.len() - 1
    }

    /// Compute the free energy F = -kT·ln(Z).
    pub fn free_energy(&self) -> f64 {
        -self.boltzmann_constant * self.temperature * self.partition_function.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    /// Tolerance for floating-point comparisons.
    const TOL: f64 = 1e-10;

    #[test]
    fn test_probabilities_sum_to_one() {
        let dist = BoltzmannDistribution::new(vec![1.0, 2.0, 3.0], 1.0);
        let sum: f64 = dist.probabilities().iter().sum();
        assert!((sum - 1.0).abs() < TOL, "probabilities sum to {sum}");
    }

    #[test]
    fn test_uniform_energies_give_uniform_probs() {
        let dist = BoltzmannDistribution::new(vec![2.0, 2.0, 2.0, 2.0], 1.0);
        for p in dist.probabilities() {
            assert!((*p - 0.25).abs() < TOL, "expected 0.25, got {p}");
        }
    }

    #[test]
    fn test_low_temperature_concentrates_on_lowest_energy() {
        let dist = BoltzmannDistribution::new(vec![0.0, 5.0, 10.0], 0.01);
        assert!(
            dist.probabilities()[0] > 0.99,
            "lowest energy should dominate at low T"
        );
    }

    #[test]
    fn test_high_temperature_uniform() {
        let dist = BoltzmannDistribution::new(vec![0.0, 1.0, 2.0], 1000.0);
        for p in dist.probabilities() {
            assert!((*p - 1.0 / 3.0).abs() < 0.01, "nearly uniform at high T");
        }
    }

    #[test]
    fn test_partition_function_two_states() {
        // E = [0, 1], T = 1, k = 1 => Z = e^0 + e^{-1} = 1 + 1/e
        let dist = BoltzmannDistribution::new(vec![0.0, 1.0], 1.0);
        let expected_z = 1.0 + 1.0 / E;
        assert!(
            (dist.partition_function() - expected_z).abs() < TOL,
            "Z = {}",
            dist.partition_function()
        );
    }

    #[test]
    fn test_mean_energy() {
        let dist = BoltzmannDistribution::new(vec![0.0, 1.0], 1.0);
        let z = 1.0 + 1.0 / E;
        let expected = (0.0 * 1.0 + 1.0 / E) / z;
        assert!(
            (dist.mean_energy() - expected).abs() < TOL,
            "mean = {}",
            dist.mean_energy()
        );
    }

    #[test]
    fn test_variance_non_negative() {
        let dist = BoltzmannDistribution::new(vec![1.0, 3.0, 5.0, 7.0], 2.0);
        assert!(dist.variance() >= -TOL, "variance should be non-negative");
    }

    #[test]
    fn test_entropy_maximum_for_uniform() {
        let dist = BoltzmannDistribution::new(vec![5.0, 5.0, 5.0], 1.0);
        // Max entropy for 3 states: ln(3)
        let expected = (3.0_f64).ln();
        assert!(
            (dist.entropy() - expected).abs() < TOL,
            "entropy = {}",
            dist.entropy()
        );
    }

    #[test]
    fn test_sample_distribution_approximately_correct() {
        let dist = BoltzmannDistribution::new(vec![0.0, 1.0], 1.0);
        let mut rng = Xorshift64::new(42);
        let n = 100_000;
        let mut counts = vec![0usize; 2];
        for _ in 0..n {
            counts[dist.sample(&mut rng)] += 1;
        }
        let p0 = counts[0] as f64 / n as f64;
        let expected_p0 = 1.0 / (1.0 + 1.0 / E);
        assert!(
            (p0 - expected_p0).abs() < 0.01,
            "p0 = {p0}, expected {expected_p0}"
        );
    }

    #[test]
    fn test_free_energy() {
        let dist = BoltzmannDistribution::new(vec![0.0, 1.0], 1.0);
        let expected_f = -1.0 * dist.partition_function().ln();
        assert!(
            (dist.free_energy() - expected_f).abs() < TOL,
            "F = {}",
            dist.free_energy()
        );
    }

    #[test]
    #[should_panic]
    fn test_empty_energies_panics() {
        BoltzmannDistribution::new(vec![], 1.0);
    }

    #[test]
    #[should_panic]
    fn test_zero_temperature_panics() {
        BoltzmannDistribution::new(vec![1.0], 0.0);
    }
}
