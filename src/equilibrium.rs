//! Equilibrium and detailed balance for agent state transitions.
//!
//! In statistical mechanics, a system at thermal equilibrium satisfies detailed balance:
//! P(i)·W(i→j) = P(j)·W(j→i), where W is the transition rate matrix.
//! This module verifies and enforces this condition for agent state models.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::distribution::BoltzmannDistribution;

/// Transition rate matrix W[i][j] = rate of transition from state i to state j.
pub type TransitionMatrix = Vec<Vec<f64>>;

/// Detailed balance verification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedBalanceResult {
    /// Whether detailed balance is satisfied for all state pairs.
    pub satisfied: bool,
    /// Maximum absolute violation of P(i)*W(i,j) - P(j)*W(j,i).
    pub max_violation: f64,
    /// Mean absolute violation across all pairs.
    pub mean_violation: f64,
    /// Per-pair violations: (i, j, |P(i)*W(i,j) - P(j)*W(j,i)|).
    pub violations: Vec<(usize, usize, f64)>,
}

/// Detailed balance checker and enforcer for agent state transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedBalance {
    /// Number of states.
    n_states: usize,
    /// Transition rate matrix W[i][j].
    transition_matrix: TransitionMatrix,
}

impl DetailedBalance {
    /// Create from an existing transition matrix.
    pub fn new(transition_matrix: TransitionMatrix) -> Self {
        let n = transition_matrix.len();
        assert!(
            transition_matrix.iter().all(|row| row.len() == n),
            "transition matrix must be square"
        );
        Self {
            n_states: n,
            transition_matrix,
        }
    }

    /// Get the number of states.
    pub fn n_states(&self) -> usize {
        self.n_states
    }

    /// Get the transition matrix.
    pub fn transition_matrix(&self) -> &TransitionMatrix {
        &self.transition_matrix
    }

    /// Verify detailed balance against a given Boltzmann distribution.
    ///
    /// Checks that P(i)·W(i→j) = P(j)·W(j→i) for all i,j.
    pub fn verify(
        &self,
        distribution: &BoltzmannDistribution,
        tolerance: f64,
    ) -> DetailedBalanceResult {
        let probs = distribution.probabilities();
        let mut violations = Vec::new();
        let mut max_violation = 0.0_f64;

        for i in 0..self.n_states {
            for j in (i + 1)..self.n_states {
                let lhs = probs[i] * self.transition_matrix[i][j];
                let rhs = probs[j] * self.transition_matrix[j][i];
                let violation = (lhs - rhs).abs();
                if violation > tolerance {
                    violations.push((i, j, violation));
                }
                max_violation = max_violation.max(violation);
            }
        }

        let n_pairs = self.n_states * (self.n_states - 1) / 2;
        let mean_violation = if n_pairs > 0 {
            let total: f64 = (0..self.n_states)
                .flat_map(|i| {
                    (i + 1..self.n_states).map(move |j| {
                        (probs[i] * self.transition_matrix[i][j]
                            - probs[j] * self.transition_matrix[j][i])
                            .abs()
                    })
                })
                .sum();
            total / n_pairs as f64
        } else {
            0.0
        };

        DetailedBalanceResult {
            satisfied: violations.is_empty(),
            max_violation,
            mean_violation,
            violations,
        }
    }

    /// Construct a transition matrix that satisfies detailed balance with the given distribution.
    ///
    /// Uses Metropolis-like transition rates: W(i→j) = min(1, P(j)/P(i)) / (n-1).
    pub fn from_distribution(distribution: &BoltzmannDistribution) -> Self {
        let probs = distribution.probabilities();
        let n = probs.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                if i != j {
                    let rate = if probs[i] > 0.0 {
                        (probs[j] / probs[i]).min(1.0) / (n - 1) as f64
                    } else {
                        1.0 / (n - 1) as f64
                    };
                    matrix[i][j] = rate;
                    row_sum += rate;
                }
            }
            // Self-loop rate = 1 - sum of off-diagonal rates
            matrix[i][i] = (1.0 - row_sum).max(0.0);
        }

        Self {
            n_states: n,
            transition_matrix: matrix,
        }
    }

    /// Compute the stationary distribution of the transition matrix via power iteration.
    pub fn stationary_distribution(&self, max_iterations: usize, tolerance: f64) -> Vec<f64> {
        let n = self.n_states;
        let mut dist = vec![1.0 / n as f64; n];

        for _ in 0..max_iterations {
            let mut new_dist = vec![0.0; n];
            for (i, nd) in new_dist.iter_mut().enumerate() {
                for (j, d) in dist.iter().enumerate() {
                    *nd += d * self.transition_matrix[j][i];
                }
            }
            // Normalize
            let sum: f64 = new_dist.iter().sum();
            if sum > 0.0 {
                for x in new_dist.iter_mut() {
                    *x /= sum;
                }
            }
            // Check convergence
            let max_change = dist
                .iter()
                .zip(new_dist.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            dist = new_dist;
            if max_change < tolerance {
                break;
            }
        }
        dist
    }
}

/// Boltzmann equilibrium: a system at thermal equilibrium with Boltzmann distribution over states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoltzmannEquilibrium {
    /// The Boltzmann distribution at equilibrium.
    distribution: BoltzmannDistribution,
    /// The detailed balance structure.
    balance: DetailedBalance,
}

impl BoltzmannEquilibrium {
    /// Create a Boltzmann equilibrium system from energies and temperature.
    pub fn new(energies: Vec<f64>, temperature: f64) -> Self {
        let distribution = BoltzmannDistribution::new(energies, temperature);
        let balance = DetailedBalance::from_distribution(&distribution);
        Self {
            distribution,
            balance,
        }
    }

    /// Get the equilibrium distribution.
    pub fn distribution(&self) -> &BoltzmannDistribution {
        &self.distribution
    }

    /// Get the detailed balance structure.
    pub fn balance(&self) -> &DetailedBalance {
        &self.balance
    }

    /// Verify equilibrium is self-consistent (transition matrix reproduces Boltzmann distribution).
    pub fn verify_self_consistency(&self, tolerance: f64) -> bool {
        let stationary = self.balance.stationary_distribution(10000, 1e-12);
        let probs = self.distribution.probabilities();
        probs
            .iter()
            .zip(stationary.iter())
            .all(|(p, s)| (p - s).abs() < tolerance)
    }

    /// Create from a map of named states to energies.
    pub fn from_named_states(
        states: HashMap<String, f64>,
        temperature: f64,
    ) -> (Self, Vec<String>) {
        let mut names: Vec<String> = states.keys().cloned().collect();
        names.sort();
        let energies: Vec<f64> = names.iter().map(|n| states[n]).collect();
        let eq = Self::new(energies, temperature);
        (eq, names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_transitions_satisfy_balance() {
        // Symmetric transition matrix with uniform energies
        let matrix = vec![
            vec![0.5, 0.25, 0.25],
            vec![0.25, 0.5, 0.25],
            vec![0.25, 0.25, 0.5],
        ];
        let db = DetailedBalance::new(matrix);
        let dist = BoltzmannDistribution::new(vec![1.0, 1.0, 1.0], 1.0);
        let result = db.verify(&dist, 1e-10);
        assert!(
            result.satisfied,
            "symmetric + uniform should satisfy balance"
        );
    }

    #[test]
    fn test_from_distribution_satisfies_balance() {
        let dist = BoltzmannDistribution::new(vec![0.0, 1.0, 2.0], 1.0);
        let db = DetailedBalance::from_distribution(&dist);
        let result = db.verify(&dist, 1e-10);
        assert!(result.satisfied, "Metropolis rates should satisfy balance");
    }

    #[test]
    fn test_violation_detection() {
        let matrix = vec![
            vec![0.9, 0.1],
            vec![0.9, 0.1], // asymmetric
        ];
        let db = DetailedBalance::new(matrix);
        let dist = BoltzmannDistribution::new(vec![1.0, 2.0], 1.0);
        let result = db.verify(&dist, 1e-10);
        // With non-uniform distribution, asymmetric matrix likely violates balance
        assert!(!result.violations.is_empty() || result.max_violation > 1e-10);
    }

    #[test]
    fn test_stationary_distribution_uniform() {
        let matrix = vec![
            vec![0.5, 0.25, 0.25],
            vec![0.25, 0.5, 0.25],
            vec![0.25, 0.25, 0.5],
        ];
        let db = DetailedBalance::new(matrix);
        let stationary = db.stationary_distribution(1000, 1e-12);
        for p in &stationary {
            assert!(
                (*p - 1.0 / 3.0).abs() < 1e-6,
                "uniform stationary: {stationary:?}"
            );
        }
    }

    #[test]
    fn test_boltzmann_equilibrium_creation() {
        let eq = BoltzmannEquilibrium::new(vec![0.0, 1.0, 2.0], 1.0);
        assert_eq!(eq.distribution().n_states(), 3);
        assert!(eq.verify_self_consistency(1e-6));
    }

    #[test]
    fn test_named_states() {
        let mut states = HashMap::new();
        states.insert("idle".to_string(), 0.0);
        states.insert("active".to_string(), 1.0);
        states.insert("busy".to_string(), 3.0);
        let (eq, names) = BoltzmannEquilibrium::from_named_states(states, 1.0);
        assert_eq!(names.len(), 3);
        assert_eq!(eq.distribution().n_states(), 3);
    }

    #[test]
    fn test_transition_matrix_rows_sum_to_one() {
        let dist = BoltzmannDistribution::new(vec![0.0, 1.0, 2.0, 3.0], 1.0);
        let db = DetailedBalance::from_distribution(&dist);
        for row in db.transition_matrix() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "row sums to {sum}, expected 1.0");
        }
    }

    #[test]
    fn test_max_violation_non_negative() {
        let matrix = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let db = DetailedBalance::new(matrix);
        let dist = BoltzmannDistribution::new(vec![0.0, 1.0], 1.0);
        let result = db.verify(&dist, 0.0);
        assert!(result.max_violation >= 0.0);
    }
}
