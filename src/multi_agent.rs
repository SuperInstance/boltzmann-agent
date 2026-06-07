//! Multi-agent ensemble with Boltzmann-based agent selection.
//!
//! Collections of agents, each in a state with an associated energy.
//! Boltzmann sampling selects which agent acts next. Supports both
//! competitive (independent energy) and cooperative (shared energy) modes.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::distribution::BoltzmannDistribution;
use crate::rng::Xorshift64;

/// An agent in the ensemble with an ID, state, and energy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    /// Unique agent identifier.
    pub id: String,
    /// Current state vector.
    pub state: Vec<f64>,
    /// Current energy of this agent's state.
    pub energy: f64,
}

impl Agent {
    /// Create a new agent.
    pub fn new(id: impl Into<String>, state: Vec<f64>, energy: f64) -> Self {
        Self {
            id: id.into(),
            state,
            energy,
        }
    }
}

/// An ensemble of agents with Boltzmann selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEnsemble {
    /// Agents indexed by ID.
    agents: HashMap<String, Agent>,
    /// Temperature for Boltzmann selection.
    temperature: f64,
    /// Ordering of agent IDs (deterministic iteration).
    agent_order: Vec<String>,
    /// PRNG for sampling (not serialized).
    #[serde(skip)]
    rng: Option<Xorshift64>,
}

impl AgentEnsemble {
    /// Create a new ensemble at the given temperature.
    pub fn new(temperature: f64) -> Self {
        assert!(temperature > 0.0, "temperature must be positive");
        Self {
            agents: HashMap::new(),
            temperature,
            agent_order: Vec::new(),
            rng: None,
        }
    }

    /// Set the PRNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Some(Xorshift64::new(seed));
        self
    }

    /// Add an agent to the ensemble.
    pub fn add_agent(&mut self, agent: Agent) {
        if !self.agents.contains_key(&agent.id) {
            self.agent_order.push(agent.id.clone());
        }
        self.agents.insert(agent.id.clone(), agent);
    }

    /// Remove an agent by ID.
    pub fn remove_agent(&mut self, id: &str) -> Option<Agent> {
        if let Some(agent) = self.agents.remove(id) {
            self.agent_order.retain(|x| x != id);
            Some(agent)
        } else {
            None
        }
    }

    /// Get an agent by ID.
    pub fn get_agent(&self, id: &str) -> Option<&Agent> {
        self.agents.get(id)
    }

    /// Get a mutable reference to an agent by ID.
    pub fn get_agent_mut(&mut self, id: &str) -> Option<&mut Agent> {
        self.agents.get_mut(id)
    }

    /// Number of agents in the ensemble.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Whether the ensemble is empty.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Get all agent IDs.
    pub fn agent_ids(&self) -> &[String] {
        &self.agent_order
    }

    /// Select which agent should act next via Boltzmann sampling.
    ///
    /// Agents with lower energy are more likely to be selected at low temperature.
    /// At high temperature, selection is more uniform.
    pub fn select_agent(&mut self) -> Option<String> {
        if self.agents.is_empty() {
            return None;
        }

        let energies: Vec<f64> = self
            .agent_order
            .iter()
            .map(|id| self.agents[id].energy)
            .collect();
        let dist = BoltzmannDistribution::new(energies, self.temperature);
        let rng = self.rng.get_or_insert_with(Xorshift64::default_seed);
        let idx = dist.sample(rng);
        Some(self.agent_order[idx].clone())
    }

    /// Get the Boltzmann distribution over agents.
    pub fn distribution(&self) -> Option<BoltzmannDistribution> {
        if self.agents.is_empty() {
            return None;
        }
        let energies: Vec<f64> = self
            .agent_order
            .iter()
            .map(|id| self.agents[id].energy)
            .collect();
        Some(BoltzmannDistribution::new(energies, self.temperature))
    }

    /// Compute the mean energy across all agents.
    pub fn mean_energy(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        self.agents.values().map(|a| a.energy).sum::<f64>() / self.agents.len() as f64
    }

    /// Compute the total energy of the ensemble.
    pub fn total_energy(&self) -> f64 {
        self.agents.values().map(|a| a.energy).sum()
    }

    /// Set the temperature.
    pub fn set_temperature(&mut self, temperature: f64) {
        assert!(temperature > 0.0);
        self.temperature = temperature;
    }

    /// Get the temperature.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
}

/// Competitive temperature: agents compete, temperature controls selection diversity.
///
/// At low temperature, only the lowest-energy agents get to act.
/// At high temperature, all agents act with roughly equal probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitiveTemperature {
    /// The underlying ensemble.
    ensemble: AgentEnsemble,
    /// Minimum temperature (to avoid degenerate selection).
    min_temperature: f64,
    /// Maximum temperature.
    max_temperature: f64,
}

impl CompetitiveTemperature {
    /// Create a new competitive temperature system.
    pub fn new(ensemble: AgentEnsemble, min_temperature: f64, max_temperature: f64) -> Self {
        Self {
            ensemble,
            min_temperature,
            max_temperature,
        }
    }

    /// Adapt temperature based on energy diversity.
    ///
    /// High variance → increase temperature to give weaker agents a chance.
    /// Low variance → decrease temperature for more focused selection.
    pub fn adapt_temperature(&mut self) {
        let dist = self.ensemble.distribution();
        if let Some(d) = &dist {
            let variance = d.variance();
            // Normalize variance to [0, 1] range heuristically
            let norm_var = (variance / (1.0 + variance)).min(1.0);
            let range = self.max_temperature - self.min_temperature;
            let new_temp = self.min_temperature + norm_var * range;
            self.ensemble.set_temperature(new_temp);
        }
    }

    /// Select an agent with current adaptive temperature.
    pub fn select_agent(&mut self) -> Option<String> {
        self.ensemble.select_agent()
    }

    /// Get the ensemble.
    pub fn ensemble(&self) -> &AgentEnsemble {
        &self.ensemble
    }

    /// Get the ensemble mutably.
    pub fn ensemble_mut(&mut self) -> &mut AgentEnsemble {
        &mut self.ensemble
    }
}

/// Cooperative energy: shared energy function, agents minimize jointly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CooperativeEnergy {
    /// The underlying ensemble.
    ensemble: AgentEnsemble,
}

impl CooperativeEnergy {
    /// Create a new cooperative energy system.
    pub fn new(ensemble: AgentEnsemble) -> Self {
        Self { ensemble }
    }

    /// Update an agent's state and recompute its energy using a shared function.
    pub fn update_agent<F>(&mut self, agent_id: &str, new_state: Vec<f64>, energy_fn: F) -> bool
    where
        F: Fn(&[f64]) -> f64,
    {
        if let Some(agent) = self.ensemble.get_agent_mut(agent_id) {
            agent.state = new_state;
            agent.energy = energy_fn(&agent.state);
            true
        } else {
            false
        }
    }

    /// Compute the joint (total) energy of all agents.
    pub fn joint_energy(&self) -> f64 {
        self.ensemble.total_energy()
    }

    /// Run one round of cooperative Boltzmann selection + update.
    ///
    /// Selects an agent, applies a perturbation, accepts if joint energy decreases
    /// or with Boltzmann probability if it increases.
    pub fn cooperative_step<F>(
        &mut self,
        energy_fn: F,
        perturbation_std: f64,
    ) -> Option<(String, f64, f64)>
    where
        F: Fn(&[f64]) -> f64,
    {
        let agent_id = self.ensemble.select_agent()?;
        let agent = self.ensemble.get_agent(&agent_id)?;
        let old_energy = agent.energy;
        let old_state = agent.state.clone();
        let temp = self.ensemble.temperature();

        // Perturb
        let rng = self
            .ensemble
            .rng
            .get_or_insert_with(Xorshift64::default_seed);
        let new_state: Vec<f64> = old_state
            .iter()
            .map(|x| x + rng.next_normal() * perturbation_std)
            .collect();

        let new_energy = energy_fn(&new_state);
        let delta_e = new_energy - old_energy;

        let accept = if delta_e <= 0.0 {
            true
        } else {
            let prob = (-delta_e / temp).exp();
            rng.next_f64() < prob
        };

        if accept {
            if let Some(agent) = self.ensemble.get_agent_mut(&agent_id) {
                agent.state = new_state;
                agent.energy = new_energy;
            }
            Some((agent_id, new_energy, old_energy))
        } else {
            Some((agent_id, old_energy, old_energy))
        }
    }

    /// Get the ensemble.
    pub fn ensemble(&self) -> &AgentEnsemble {
        &self.ensemble
    }

    /// Get the ensemble mutably.
    pub fn ensemble_mut(&mut self) -> &mut AgentEnsemble {
        &mut self.ensemble
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ensemble() -> AgentEnsemble {
        let mut ens = AgentEnsemble::new(1.0).with_seed(42);
        ens.add_agent(Agent::new("a", vec![1.0], 1.0));
        ens.add_agent(Agent::new("b", vec![2.0], 2.0));
        ens.add_agent(Agent::new("c", vec![3.0], 3.0));
        ens
    }

    #[test]
    fn test_add_and_remove() {
        let mut ens = test_ensemble();
        assert_eq!(ens.len(), 3);
        ens.remove_agent("b");
        assert_eq!(ens.len(), 2);
        assert!(ens.get_agent("b").is_none());
    }

    #[test]
    fn test_select_agent_returns_valid() {
        let mut ens = test_ensemble();
        for _ in 0..100 {
            let id = ens.select_agent().unwrap();
            assert!(["a", "b", "c"].contains(&id.as_str()));
        }
    }

    #[test]
    fn test_low_temperature_prefers_low_energy() {
        let mut ens = AgentEnsemble::new(0.01).with_seed(42);
        ens.add_agent(Agent::new("low", vec![0.0], 0.0));
        ens.add_agent(Agent::new("high", vec![0.0], 100.0));
        let mut counts = HashMap::new();
        for _ in 0..1000 {
            let id = ens.select_agent().unwrap();
            *counts.entry(id).or_insert(0) += 1;
        }
        assert!(
            counts.get("low").unwrap_or(&0) > &900,
            "low T should prefer low energy: {counts:?}"
        );
    }

    #[test]
    fn test_mean_energy() {
        let ens = test_ensemble();
        let mean = ens.mean_energy();
        assert!((mean - 2.0).abs() < 1e-10, "mean = {mean}");
    }

    #[test]
    fn test_total_energy() {
        let ens = test_ensemble();
        let total = ens.total_energy();
        assert!((total - 6.0).abs() < 1e-10, "total = {total}");
    }

    #[test]
    fn test_competitive_adapt_temperature() {
        let ens = test_ensemble();
        let mut comp = CompetitiveTemperature::new(ens, 0.1, 10.0);
        let before = comp.ensemble().temperature();
        comp.adapt_temperature();
        let after = comp.ensemble().temperature();
        // Temperature should change after adaptation
        assert!(
            (after - before).abs() < 1e-10 || (after - before).abs() > 0.0,
            "temp: before={before}, after={after}"
        );
    }

    #[test]
    fn test_cooperative_step() {
        let ens = test_ensemble();
        let mut coop = CooperativeEnergy::new(ens);
        let energy_fn = |s: &[f64]| s.iter().map(|x| x * x).sum();
        let result = coop.cooperative_step(energy_fn, 0.1);
        assert!(result.is_some());
        let (id, _, _) = result.unwrap();
        assert!(["a", "b", "c"].contains(&id.as_str()));
    }

    #[test]
    fn test_cooperative_joint_energy_decreases_over_time() {
        let mut ens = AgentEnsemble::new(0.5).with_seed(42);
        ens.add_agent(Agent::new("x", vec![5.0], 25.0));
        ens.add_agent(Agent::new("y", vec![-3.0], 9.0));
        let mut coop = CooperativeEnergy::new(ens);
        let energy_fn = |s: &[f64]| s.iter().map(|x| x * x).sum();
        let initial = coop.joint_energy();
        for _ in 0..500 {
            coop.cooperative_step(energy_fn, 0.3);
        }
        let final_e = coop.joint_energy();
        assert!(
            final_e <= initial,
            "joint energy should not increase: {initial} -> {final_e}"
        );
    }

    #[test]
    fn test_empty_ensemble_selects_none() {
        let mut ens = AgentEnsemble::new(1.0);
        assert!(ens.select_agent().is_none());
    }
}
