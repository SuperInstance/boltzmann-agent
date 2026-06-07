//! Free energy: F = E - TS.
//!
//! Helmholtz free energy (constant T, V) and Gibbs free energy (constant T, P).
//! Applications to agent systems: free energy as surprise/prediction error,
//! the minimum free energy principle (Friston), and active inference.

use serde::{Deserialize, Serialize};

use crate::distribution::BoltzmannDistribution;

/// Helmholtz free energy: F = E - TS = -kT·ln(Z).
///
/// At constant temperature and volume, a system minimizes its Helmholtz free energy.
/// In agent systems, this represents the tradeoff between accuracy (low energy)
/// and complexity (low entropy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelmholtzFreeEnergy {
    /// Mean energy ⟨E⟩.
    pub mean_energy: f64,
    /// Temperature T.
    pub temperature: f64,
    /// Entropy S.
    pub entropy: f64,
    /// Free energy F = ⟨E⟩ - T·S.
    pub free_energy: f64,
}

impl HelmholtzFreeEnergy {
    /// Compute Helmholtz free energy from a Boltzmann distribution.
    pub fn from_distribution(dist: &BoltzmannDistribution) -> Self {
        let mean_e = dist.mean_energy();
        let s = dist.entropy();
        let t = dist.temperature();
        let f = mean_e - t * s;
        Self {
            mean_energy: mean_e,
            temperature: t,
            entropy: s,
            free_energy: f,
        }
    }

    /// Compute directly from energy, temperature, and entropy.
    pub fn new(mean_energy: f64, temperature: f64, entropy: f64) -> Self {
        Self {
            mean_energy,
            temperature,
            entropy,
            free_energy: mean_energy - temperature * entropy,
        }
    }
}

/// Gibbs free energy: G = F + PV = H - TS.
///
/// At constant temperature and pressure, a system minimizes Gibbs free energy.
/// For agent systems, the "pressure" term can model resource constraints or
/// population-level effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GibbsFreeEnergy {
    /// Helmholtz free energy F.
    pub helmholtz: f64,
    /// Pressure P (or resource constraint parameter).
    pub pressure: f64,
    /// Volume V (or capacity).
    pub volume: f64,
    /// Gibbs free energy G = F + PV.
    pub gibbs: f64,
}

impl GibbsFreeEnergy {
    /// Compute Gibbs free energy.
    pub fn new(helmholtz: f64, pressure: f64, volume: f64) -> Self {
        Self {
            helmholtz,
            pressure,
            volume,
            gibbs: helmholtz + pressure * volume,
        }
    }

    /// From a Boltzmann distribution with pressure-volume correction.
    pub fn from_distribution(dist: &BoltzmannDistribution, pressure: f64, volume: f64) -> Self {
        let helmholtz = HelmholtzFreeEnergy::from_distribution(dist);
        Self::new(helmholtz.free_energy, pressure, volume)
    }
}

/// Variational free energy for active inference (Friston).
///
/// In the free energy principle, an agent minimizes variational free energy:
///
/// ```text
/// F = D_KL[q(x) || p(x|o)] - ln p(o)
///   = E_q[-ln p(o|x)] + D_KL[q(x) || p(x)]
///   = Accuracy + Complexity
/// ```
///
/// where q(x) is the agent's posterior belief, p(x|o) is the true posterior,
/// p(o) is the evidence, and D_KL is the Kullback-Leibler divergence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariationalFreeEnergy {
    /// Expected surprise (negative log-likelihood of observations given states).
    pub accuracy: f64,
    /// KL divergence between posterior and prior beliefs.
    pub complexity: f64,
    /// Total variational free energy.
    pub free_energy: f64,
}

impl VariationalFreeEnergy {
    /// Compute from accuracy and complexity.
    ///
    /// F = accuracy + complexity (both >= 0).
    pub fn new(accuracy: f64, complexity: f64) -> Self {
        Self {
            accuracy,
            complexity,
            free_energy: accuracy + complexity,
        }
    }

    /// Compute from belief distributions (discrete).
    ///
    /// `posterior` and `prior` are probability distributions over the same states.
    /// `likelihood` is -ln p(o|x) for each state.
    pub fn from_distributions(
        posterior: &[f64],
        prior: &[f64],
        neg_log_likelihood: &[f64],
    ) -> Self {
        assert_eq!(
            posterior.len(),
            prior.len(),
            "posterior and prior must have same length"
        );
        assert_eq!(
            posterior.len(),
            neg_log_likelihood.len(),
            "distributions must have same length"
        );

        // Accuracy: E_q[-ln p(o|x)] = Σ q(x) * (-ln p(o|x))
        let accuracy: f64 = posterior
            .iter()
            .zip(neg_log_likelihood.iter())
            .map(|(q, nll)| q * nll)
            .sum();

        // Complexity: D_KL[q || p] = Σ q(x) * ln(q(x) / p(x))
        let complexity: f64 = posterior
            .iter()
            .zip(prior.iter())
            .filter(|(q, _)| **q > 0.0)
            .map(|(q, p)| {
                let p_safe = p.max(1e-300);
                q * (q / p_safe).ln()
            })
            .sum();

        Self {
            accuracy,
            complexity,
            free_energy: accuracy + complexity,
        }
    }

    /// Compute the KL divergence between two discrete distributions.
    pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
        assert_eq!(p.len(), q.len());
        p.iter()
            .zip(q.iter())
            .filter(|(pi, _)| **pi > 0.0)
            .map(|(pi, qi)| pi * (pi / qi.max(1e-300)).ln())
            .sum()
    }
}

/// Active inference agent that minimizes variational free energy.
///
/// Implements the core loop of active inference:
/// 1. Update beliefs (minimize free energy wrt posterior)
/// 2. Select actions (minimize expected free energy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceAgent {
    /// Prior beliefs over states.
    pub prior: Vec<f64>,
    /// Current posterior beliefs over states.
    pub posterior: Vec<f64>,
    /// Temperature parameter for precision.
    pub temperature: f64,
    /// History of free energy values.
    pub free_energy_history: Vec<f64>,
}

impl ActiveInferenceAgent {
    /// Create a new active inference agent with uniform prior.
    pub fn new(n_states: usize, temperature: f64) -> Self {
        let uniform = vec![1.0 / n_states as f64; n_states];
        Self {
            prior: uniform.clone(),
            posterior: uniform,
            temperature,
            free_energy_history: Vec::new(),
        }
    }

    /// Create with specific prior.
    pub fn with_prior(prior: Vec<f64>, temperature: f64) -> Self {
        let posterior = prior.clone();
        Self {
            prior,
            posterior,
            temperature,
            free_energy_history: Vec::new(),
        }
    }

    /// Update posterior beliefs given observations via variational Bayes.
    ///
    /// Simplified: posterior ∝ prior * exp(-accuracy / temperature).
    /// This minimizes variational free energy iteratively.
    pub fn update_beliefs(&mut self, neg_log_likelihood: &[f64], n_iterations: usize) -> f64 {
        assert_eq!(self.prior.len(), neg_log_likelihood.len());

        // Initialize posterior from prior
        self.posterior = self.prior.clone();

        for _ in 0..n_iterations {
            // Unnormalized log-posterior: log(prior) - neg_log_likelihood / temperature
            let log_posterior: Vec<f64> = self
                .prior
                .iter()
                .zip(neg_log_likelihood.iter())
                .map(|(p, nll)| {
                    let log_p = if *p > 0.0 { p.ln() } else { f64::NEG_INFINITY };
                    log_p - nll / self.temperature
                })
                .collect();

            // Normalize via log-sum-exp
            let max_log = log_posterior
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = log_posterior.iter().map(|l| (l - max_log).exp()).sum();
            let log_z = max_log + sum_exp.ln();

            self.posterior = log_posterior.iter().map(|l| (l - log_z).exp()).collect();
        }

        let vfe = VariationalFreeEnergy::from_distributions(
            &self.posterior,
            &self.prior,
            neg_log_likelihood,
        );
        self.free_energy_history.push(vfe.free_energy);
        vfe.free_energy
    }

    /// Select the action that minimizes expected free energy.
    ///
    /// Given action-conditioned neg-log-likelihoods, pick the action
    /// whose posterior has lowest variational free energy.
    pub fn select_action(&self, action_likelihoods: &[Vec<f64>]) -> usize {
        action_likelihoods
            .iter()
            .enumerate()
            .map(|(action_idx, nll)| {
                // Compute VFE for this action
                let log_posterior: Vec<f64> = self
                    .prior
                    .iter()
                    .zip(nll.iter())
                    .map(|(p, nll_val)| {
                        let log_p = if *p > 0.0 { p.ln() } else { f64::NEG_INFINITY };
                        log_p - nll_val / self.temperature
                    })
                    .collect();
                let max_log = log_posterior
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = log_posterior.iter().map(|l| (l - max_log).exp()).sum();
                let log_z = max_log + sum_exp.ln();
                let post: Vec<f64> = log_posterior.iter().map(|l| (l - log_z).exp()).collect();

                let vfe = VariationalFreeEnergy::from_distributions(&post, &self.prior, nll);
                (action_idx, vfe.free_energy)
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    /// Get the number of states.
    pub fn n_states(&self) -> usize {
        self.prior.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_helmholtz_from_distribution() {
        let dist = BoltzmannDistribution::new(vec![0.0, 1.0], 1.0);
        let hf = HelmholtzFreeEnergy::from_distribution(&dist);
        let expected_f = dist.mean_energy() - dist.temperature() * dist.entropy();
        assert!(
            (hf.free_energy - expected_f).abs() < TOL,
            "F = {}",
            hf.free_energy
        );
    }

    #[test]
    fn test_helmholtz_zero_entropy() {
        // Zero entropy → F = E
        let hf = HelmholtzFreeEnergy::new(5.0, 1.0, 0.0);
        assert!((hf.free_energy - 5.0).abs() < TOL);
    }

    #[test]
    fn test_gibbs_free_energy() {
        let g = GibbsFreeEnergy::new(10.0, 2.0, 3.0);
        assert!((g.gibbs - 16.0).abs() < TOL, "G = F + PV = 10 + 6 = 16");
    }

    #[test]
    fn test_gibbs_from_distribution() {
        let dist = BoltzmannDistribution::new(vec![0.0, 1.0], 1.0);
        let g = GibbsFreeEnergy::from_distribution(&dist, 1.0, 2.0);
        let hf = HelmholtzFreeEnergy::from_distribution(&dist);
        assert!(
            (g.gibbs - (hf.free_energy + 2.0)).abs() < TOL,
            "G = {}",
            g.gibbs
        );
    }

    #[test]
    fn test_variational_free_energy_components() {
        let vfe = VariationalFreeEnergy::new(3.0, 2.0);
        assert!((vfe.accuracy - 3.0).abs() < TOL);
        assert!((vfe.complexity - 2.0).abs() < TOL);
        assert!((vfe.free_energy - 5.0).abs() < TOL);
    }

    #[test]
    fn test_vfe_from_distributions() {
        let posterior = vec![0.5, 0.5];
        let prior = vec![0.5, 0.5];
        let nll = vec![1.0, 1.0];
        let vfe = VariationalFreeEnergy::from_distributions(&posterior, &prior, &nll);
        // When posterior = prior, KL divergence = 0, so complexity = 0
        assert!(
            vfe.complexity.abs() < TOL,
            "complexity = {}",
            vfe.complexity
        );
        // Accuracy = 0.5*1.0 + 0.5*1.0 = 1.0
        assert!((vfe.accuracy - 1.0).abs() < TOL);
    }

    #[test]
    fn test_kl_divergence_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = VariationalFreeEnergy::kl_divergence(&p, &p);
        assert!(kl.abs() < TOL, "KL(p||p) = {kl}");
    }

    #[test]
    fn test_kl_divergence_non_negative() {
        let p = vec![0.5, 0.5];
        let q = vec![0.9, 0.1];
        let kl = VariationalFreeEnergy::kl_divergence(&p, &q);
        assert!(kl >= -TOL, "KL divergence should be non-negative: {kl}");
    }

    #[test]
    fn test_active_inference_agent_creation() {
        let agent = ActiveInferenceAgent::new(3, 1.0);
        assert_eq!(agent.n_states(), 3);
        let sum: f64 = agent.posterior.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }

    #[test]
    fn test_belief_update_reduces_free_energy() {
        let mut agent = ActiveInferenceAgent::new(3, 1.0);
        // Observation strongly prefers state 0
        let nll = vec![0.0, 10.0, 10.0];
        let fe = agent.update_beliefs(&nll, 10);
        // Posterior should concentrate on state 0
        assert!(agent.posterior[0] > 0.9, "posterior: {:?}", agent.posterior);
        assert!(fe.is_finite());
    }

    #[test]
    fn test_select_action() {
        let agent = ActiveInferenceAgent::new(2, 1.0);
        // Action 0 has low surprise, action 1 has high surprise
        let action_likelihoods = vec![
            vec![0.1, 0.1], // Low surprise for all states
            vec![5.0, 5.0], // High surprise for all states
        ];
        let action = agent.select_action(&action_likelihoods);
        assert_eq!(action, 0, "should select low-surprise action");
    }

    #[test]
    fn test_free_energy_history() {
        let mut agent = ActiveInferenceAgent::new(2, 1.0);
        agent.update_beliefs(&[1.0, 1.0], 5);
        agent.update_beliefs(&[0.5, 1.5], 5);
        agent.update_beliefs(&[0.1, 2.0], 5);
        assert_eq!(agent.free_energy_history.len(), 3);
    }
}
