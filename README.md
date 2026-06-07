# boltzmann-agent

**Boltzmann distribution applied to agent action selection and multi-agent systems.**

A zero-dependency (except `serde`) Rust crate that brings the mathematical framework of statistical mechanics to computational agents. Temperature-controlled exploration/exploitation, simulated annealing optimization, detailed balance verification, multi-agent ensembles, and Friston's free energy principle—all in clean, well-documented Rust.

```
[dependencies]
boltzmann-agent = "0.1"
```

---

## Table of Contents

- [Theory](#theory)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Examples](#examples)
  - [Basic Boltzmann Distribution](#example-1-basic-boltzmann-distribution-computation)
  - [Action Selection with Temperature Annealing](#example-2-action-selection-with-temperature-annealing)
  - [Simulated Annealing Optimization](#example-3-simulated-annealing-optimization)
  - [Multi-Agent Ensemble Equilibrium](#example-4-multi-agent-ensemble-equilibrium)
- [Module Reference](#module-reference)
- [Design Decisions](#design-decisions)
- [Performance](#performance)
- [Comparison with Alternatives](#comparison-with-alternatives)
- [Glossary](#glossary)
- [References](#references)
- [License](#license)

---

## Theory

### The Boltzmann Distribution

The fundamental insight connecting statistical mechanics to agent systems is the **Boltzmann distribution** (also called the Gibbs distribution). For a system with discrete states {s₁, s₂, ..., sₙ}, each with energy E(sᵢ), the probability of observing the system in state sᵢ at temperature T is:

```
P(sᵢ) = exp(-E(sᵢ) / kT) / Z
```

where:

- **β = 1/(kT)** is the **inverse temperature** (higher β → more concentrated on low-energy states)
- **k** is Boltzmann's constant (set to 1 in natural units)
- **Z = Σᵢ exp(-E(sᵢ) / kT)** is the **partition function**, the normalization constant

This is the same distribution as the **softmax** function used in machine learning, but with physical interpretation: energies are costs, temperature controls exploration, and the partition function ensures valid probabilities.

### The Partition Function

The **partition function** Z is the single most important quantity in equilibrium statistical mechanics:

```
Z = Σᵢ exp(-βE(sᵢ))
```

From Z, every thermodynamic quantity can be derived:

| Quantity | Formula |
|----------|---------|
| Mean energy | ⟨E⟩ = -∂(ln Z)/∂β |
| Entropy | S = k·ln Z + ⟨E⟩/T = k(1 + ln Z - β⟨E⟩) |
| Free energy | F = -kT·ln Z |
| Specific heat | Cᵥ = ∂⟨E⟩/∂T = β²·Var(E)/k |

### Free Energy

**Helmholtz free energy** at constant temperature and volume:

```
F = ⟨E⟩ - TS = -kT·ln Z
```

This is the quantity that systems at equilibrium minimize. For agents, F captures the tradeoff between being in low-energy (accurate) states and maintaining high entropy (exploration):

- **Low F** → agent is both accurate *and* uncertain (exploring efficiently)
- **F = ⟨E⟩** when S = 0 (agent is certain but possibly wrong)
- **F → -kT·ln(N)** when all N states are equally likely

**Gibbs free energy** adds a pressure-volume term:

```
G = F + PV
```

For agents, P can model resource constraints and V can model capacity.

### Variational Free Energy and Active Inference

Friston's **free energy principle** (Friston, 2010) posits that biological agents minimize **variational free energy**:

```
F = E_q[-ln p(o|s)] + D_KL[q(s) || p(s)]
  = Accuracy        + Complexity
```

where:
- **q(s)** is the agent's posterior belief about hidden states s
- **p(s)** is the prior
- **p(o|s)** is the likelihood of observations o given states
- **D_KL** is the Kullback-Leibler divergence

Active inference agents select actions that minimize *expected* free energy, naturally balancing exploration (reducing complexity/uncertainty) and exploitation (reducing accuracy/surprise).

### Detailed Balance

For a Markov chain with transition rates W(i→j) to have the Boltzmann distribution as its stationary distribution, it must satisfy **detailed balance**:

```
P(i)·W(i→j) = P(j)·W(j→i)    for all i, j
```

This is the **principle of microscopic reversibility**: at equilibrium, every forward transition is balanced by the corresponding reverse transition, weighted by the state probabilities. The Metropolis-Hastings algorithm constructs transition matrices that satisfy this condition.

### Temperature and Exploration-Exploitation

The temperature parameter T controls a fundamental tradeoff:

| Temperature | Behavior | Agent Analog |
|------------|----------|-------------|
| T → 0 | Always pick lowest-energy state | Pure exploitation (greedy) |
| T → ∞ | Uniform random selection | Pure exploration |
| T ~ optimal | Boltzmann-weighted selection | Balanced explore/exploit |

**Annealing schedules** gradually reduce T:
- **Linear**: T(t) = T₀ - αt — simple, predictable
- **Exponential**: T(t) = T₀·γᵗ — fast initial decay, fine-tuning later
- **Cosine**: T(t) = T_min + ½(T₀ - T_min)(1 + cos(πt/T_end)) — smooth, based on cosine schedule from SGDR (Loshchilov & Hutter, 2017)

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │     boltzmann-agent           │
                    └──────────┬──────────────────-─┘
                               │
        ┌──────────┬───────────┼───────────┬──────────────┐
        │          │           │           │              │
   ┌────▼───┐ ┌───▼────┐ ┌───▼─────┐ ┌───▼──────┐ ┌────▼────────┐
   │ dist-  │ │ selec- │ │ anneal- │ │ equili-  │ │ free_       │
   │ ribu-  │ │ tion   │ │  ing    │ │ brium    │ │ energy      │
   │ tion   │ │        │ │         │ │          │ │             │
   └────┬───┘ └───┬────┘ └────┬────┘ └────┬─────┘ └────┬────────┘
        │         │           │           │             │
        │   ┌─────▼───────┐   │     ┌─────▼────┐  ┌────▼─────┐
        │   │ Temperature │   │     │ Detailed │  │ Helmholtz│
        │   │ Schedule    │   │     │ Balance  │  │ Gibbs    │
        │   │ (Linear/    │   │     │ Boltzmann│  │ Variat.  │
        │   │  Exp/       │   │     │ Equilib. │  │ Active   │
        │   │  Cosine)    │   │     └──────────┘  │ Inference│
        │   └─────────────┘   │                   └──────────┘
        │                     │
   ┌────▼─────────────────────▼──────┐
   │          Xorshift64 PRNG        │
   │   (no external rand dependency) │
   └────────────────────────────────┘

                    Multi-Agent Ensemble
    ┌──────────────────────────────────────────┐
    │                                          │
    │  Agent A ──── Agent B ──── Agent C       │
    │  E=0.5       E=1.2       E=3.0          │
    │   ╲            ╲            ╱             │
    │    ╲            ╲          ╱              │
    │     ╲            ╲        ╱               │
    │      ─────────────────────                │
    │       Boltzmann Selection                 │
    │       P(A) > P(B) > P(C)                 │
    │       (lower energy = higher prob)        │
    │                                          │
    │  Energy Landscape:                        │
    │                                          │
    │  Energy                                   │
    │   4 ┤                                     │
    │   3 ┤          • Agent C                  │
    │   2 ┤                                     │
    │   1 ┤    • Agent B                        │
    │   0 ┤ • Agent A                           │
    │     └──────────────────── States          │
    │                                          │
    │  T=0.1: A dominates (exploit)            │
    │  T=10.: Uniform (explore)                │
    └──────────────────────────────────────────┘
```

---

## Quick Start

```rust
use boltzmann_agent::distribution::BoltzmannDistribution;
use boltzmann_agent::rng::Xorshift64;

fn main() {
    // Define energies for 4 states
    let energies = vec![1.0, 2.0, 3.0, 4.0];
    let temperature = 1.0;

    let dist = BoltzmannDistribution::new(energies, temperature);

    println!("Partition function Z = {}", dist.partition_function());
    println!("Mean energy ⟨E⟩ = {}", dist.mean_energy());
    println!("Entropy S = {}", dist.entropy());
    println!("Probabilities: {:?}", dist.probabilities());

    // Sample states
    let mut rng = Xorshift64::new(42);
    for _ in 0..10 {
        let state = dist.sample(&mut rng);
        println!("Sampled state: {state}");
    }
}
```

---

## Examples

### Example 1: Basic Boltzmann Distribution Computation

Compute thermodynamic quantities for a 5-state system at various temperatures:

```rust
use boltzmann_agent::distribution::BoltzmannDistribution;
use boltzmann_agent::rng::Xorshift64;

fn main() {
    // Define a 5-state energy landscape
    // States represent different configurations of an agent
    let energies = vec![0.0, 1.0, 2.0, 5.0, 10.0];

    // Compare behavior at different temperatures
    let temperatures = [0.1, 0.5, 1.0, 5.0, 50.0];

    println!("T      | Z        | ⟨E⟩     | Var(E)  | S       | F");
    println!("-------|----------|---------|---------|---------|--------");

    for &t in &temperatures {
        let dist = BoltzmannDistribution::new(energies.clone(), t);

        println!(
            "{:<6.1} | {:<8.4} | {:<7.4} | {:<7.4} | {:<7.4} | {:<7.4}",
            t,
            dist.partition_function(),
            dist.mean_energy(),
            dist.variance(),
            dist.entropy(),
            dist.free_energy()
        );
    }

    // At T=0.1: nearly all probability on state 0 (energy=0)
    let cold = BoltzmannDistribution::new(energies.clone(), 0.1);
    assert!(cold.probabilities()[0] > 0.99);

    // At T=50: nearly uniform distribution
    let hot = BoltzmannDistribution::new(energies.clone(), 50.0);
    let max_prob = hot.probabilities().iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(max_prob < 0.25); // No state dominates

    // Verify entropy increases with temperature
    let s_low = BoltzmannDistribution::new(energies.clone(), 0.5).entropy();
    let s_high = BoltzmannDistribution::new(energies.clone(), 10.0).entropy();
    assert!(s_high > s_low);
}
```

### Example 2: Action Selection with Temperature Annealing

An agent selecting from 5 actions with different costs, gradually shifting from exploration to exploitation:

```rust
use boltzmann_agent::selection::{ActionSelector, TemperatureSchedule};
use std::collections::HashMap;

fn main() {
    // 5 actions with associated energy costs
    let action_names = ["explore", "gather", "build", "trade", "rest"];
    let action_energies = [2.0, 1.5, 3.0, 1.0, 0.5]; // rest is cheapest

    // Anneal from exploration (T=10) to exploitation (T=0.01) over 1000 steps
    let schedule = TemperatureSchedule::linear(10.0, 0.01, 1000);
    let mut selector = ActionSelector::new(schedule).with_seed(42);

    let mut action_counts: HashMap<&str, usize> = HashMap::new();

    println!("Step | T       | Selected  | Probs");
    println!("-----|---------|-----------|------");

    for step in 0..200 {
        let idx = selector.select(&action_energies);
        let action = action_names[idx];
        *action_counts.entry(action).or_insert(0) += 1;

        if step % 40 == 0 {
            let probs = selector.probabilities(&action_energies);
            println!(
                "{:<4} | T={:<5.3} | {:<9} | {:?}",
                step, selector.temperature(), action,
                probs.iter().map(|p| format!("{p:.3}")).collect::<Vec<_>>()
            );
        }
    }

    println!("\nAction counts after 200 steps:");
    for (name, count) in &action_counts {
        println!("  {name}: {count}");
    }

    // At low temperature, "rest" (lowest energy) should dominate
    let rest_count = action_counts.get("rest").unwrap_or(&0);
    println!("Rest selected {rest_count} times (expected high at low T)");
}
```

### Example 3: Simulated Annealing Optimization

Minimize a multi-modal Rastrigin function using simulated annealing:

```rust
use boltzmann_agent::annealing::{SimulatedAnnealing, AnnealingConfig};
use boltzmann_agent::selection::TemperatureSchedule;

// Rastrigin function: many local minima, global minimum at origin with f(0) = 0
fn rastrigin(state: &[f64]) -> f64 {
    const A: f64 = 10.0;
    state.iter().map(|x| x * x - A * (2.0 * std::f64::consts::PI * x).cos() + A).sum()
}

fn main() {
    let config = AnnealingConfig {
        schedule: TemperatureSchedule::linear(50.0, 0.001, 100_000),
        max_iterations: 100_000,
        tolerance: 1e-12,
        patience: 10_000,
        proposal_std: 0.5,
        log_interval: 10_000,
    };

    let mut sa = SimulatedAnnealing::new(config).with_seed(42);

    // Start from a distant point
    let initial = vec![5.0, -4.0, 3.0];
    println!("Initial state: {initial:?}");
    println!("Initial energy: {}", rastrigin(&initial));

    let result = sa.minimize(rastrigin, initial);

    println!("\nResult:");
    println!("  Best state: {:?}", result.best_state);
    println!("  Best energy: {:.6}", result.best_energy);
    println!("  Final energy: {:.6}", result.final_energy);
    println!("  Converged: {}", result.converged);
    println!("  Energy history: {:?}", result.energy_history);

    // Should find a solution near zero energy
    assert!(result.best_energy < 1.0, "Annealing should find near-global minimum");
}
```

### Example 4: Multi-Agent Ensemble Equilibrium

Three agents competing for action selection in a shared environment:

```rust
use boltzmann_agent::multi_agent::{AgentEnsemble, Agent, CompetitiveTemperature, CooperativeEnergy};
use boltzmann_agent::equilibrium::BoltzmannEquilibrium;
use std::collections::HashMap;

fn main() {
    // Create agents with different energy states
    let mut ensemble = AgentEnsemble::new(1.0);
    ensemble.add_agent(Agent::new("scout", vec![2.0, 0.0], 2.0));
    ensemble.add_agent(Agent::new("worker", vec![0.5, 0.5], 0.5));
    ensemble.add_agent(Agent::new("leader", vec![1.0, 1.0], 1.0));

    println!("=== Agent Ensemble ===");
    println!("Mean energy: {:.3}", ensemble.mean_energy());
    println!("Total energy: {:.3}", ensemble.total_energy());

    // Select agents via Boltzmann sampling
    println!("\nAgent selection distribution:");
    let dist = ensemble.distribution().unwrap();
    for (i, id) in ensemble.agent_ids().iter().enumerate() {
        println!("  {id}: P = {:.4}, E = {:.2}",
            dist.probabilities()[i],
            ensemble.get_agent(id).unwrap().energy
        );
    }

    // Verify equilibrium
    let energies: Vec<f64> = ensemble.agent_ids()
        .iter()
        .map(|id| ensemble.get_agent(id).unwrap().energy)
        .collect();
    let eq = BoltzmannEquilibrium::new(energies, 1.0);
    println!("\nEquilibrium self-consistent: {}", eq.verify_self_consistency(1e-6));

    // Cooperative optimization: all agents minimize a shared energy
    let mut coop = CooperativeEnergy::new(ensemble);
    let energy_fn = |s: &[f64]| s.iter().map(|x| x * x).sum();

    println!("\nCooperative optimization:");
    println!("  Initial joint energy: {:.3}", coop.joint_energy());

    for round in 0..100 {
        if let Some((agent_id, new_e, old_e)) = coop.cooperative_step(energy_fn, 0.2) {
            if round % 20 == 0 {
                println!("  Round {round}: {agent_id} E={old_e:.3}→{new_e:.3}");
            }
        }
    }
    println!("  Final joint energy: {:.3}", coop.joint_energy());
}
```

---

## Module Reference

| Module | Description | Key Types |
|--------|-------------|-----------|
| `distribution` | Boltzmann distribution P(s) = exp(-βE)/Z | `BoltzmannDistribution` |
| `selection` | Action selection with annealing schedules | `ActionSelector`, `TemperatureSchedule` |
| `annealing` | Simulated annealing optimization | `SimulatedAnnealing`, `AnnealingResult`, `AnnealingConfig` |
| `equilibrium` | Detailed balance and Boltzmann equilibrium | `DetailedBalance`, `BoltzmannEquilibrium` |
| `multi_agent` | Multi-agent ensembles with competitive/cooperative modes | `AgentEnsemble`, `CompetitiveTemperature`, `CooperativeEnergy` |
| `free_energy` | Helmholtz, Gibbs, and variational free energy | `HelmholtzFreeEnergy`, `GibbsFreeEnergy`, `VariationalFreeEnergy`, `ActiveInferenceAgent` |

---

## Design Decisions

### Why a simple PRNG instead of `rand`?

The `rand` crate is excellent but brings a large dependency tree (50+ crates including `getrandom`, `libc`, etc.). For Monte Carlo sampling in agent systems, we need:
- **Reproducibility**: deterministic seeds for debugging and testing
- **Speed**: xorshift64 is extremely fast (3 XOR-shift operations)
- **No OS dependencies**: works in `no_std`-adjacent environments

The `Xorshift64` implementation passes standard statistical tests for our use case (Boltzmann sampling, proposal generation). It is **not** cryptographically secure—and it doesn't need to be.

### Why `f64` everywhere?

1. **Simplicity**: No generic bounds cluttering every function signature
2. **Performance**: `f64` maps to hardware doubles on all modern platforms
3. **Precision**: For the energy scales typical in agent systems (0.01–1000), `f64` provides 15+ significant digits
4. **Interoperability**: All physics and ML code uses `f64`

If you need `f32` for GPU computation or embedded systems, the types are simple enough to adapt.

### Why `Vec<f64>` for states instead of generic `ArrayLike`?

Generic collections add complexity that doesn't serve the crate's purpose. The mathematical operations we need (element-wise arithmetic, dot products, perturbation) are trivial with `Vec<f64>`. If you need `ndarray` or `nalgebra` integration, convert at the boundary.

### Why iterative annealing, not recursive?

1. **No stack overflow**: recursive SA on large problems can blow the stack
2. **Convergence tracking**: iterative loops naturally support patience-based stopping
3. **Debugging**: easier to inspect state at each iteration
4. **Control flow**: cleaner integration with early termination and logging

### Why only `serde` as a dependency?

All public types derive `Serialize + Deserialize` for:
- Saving/loading agent states
- Network transmission of ensemble configurations
- Configuration files for annealing runs
- Checkpoint/resume for long-running simulations

`serde` is the de facto standard for Rust serialization and most projects already depend on it.

---

## Performance

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|-----------------|-------|
| Create distribution | O(n) | O(n) | n = number of states |
| Compute Z | O(n) | O(1) | with log-sum-exp stability |
| Sample from distribution | O(n) | O(1) | cumulative sum search |
| Mean energy, variance, entropy | O(n) | O(1) | single pass |
| One SA iteration | O(d) | O(d) | d = state dimensionality |
| Full SA run (k iterations) | O(kd) | O(d) | d = dimension, k = iterations |
| Verify detailed balance | O(n²) | O(n²) | all pairs of states |
| Stationary distribution | O(mn²) | O(n) | m = iterations for power method |
| Select agent from ensemble | O(a) | O(1) | a = number of agents |
| VFE computation | O(n) | O(1) | n = number of states |
| Active inference belief update | O(in) | O(n) | i = iterations, n = states |

### Scaling with Agent Count

For an ensemble of **a** agents:

- **Selection**: O(a) to build distribution + O(a) to sample = O(a)
- **Adding/removing agents**: O(1) amortized (HashMap)
- **Cooperative step**: O(a) for selection + O(d) for perturbation = O(a + d)
- **Full cooperative round** (all agents): O(a² + ad)

For **100 agents** in a **10-dimensional** state space, expect:
- ~1μs per selection
- ~1μs per cooperative step
- ~100μs for a full cooperative round

All measurements are wall-clock estimates on a modern x86-64 system.

---

## Comparison with Alternatives

### Boltzmann vs Softmax

| Aspect | Boltzmann Selection | Softmax |
|--------|-------------------|---------|
| Formula | P(i) = exp(-Eᵢ/kT) / Z | P(i) = exp(θᵢ) / Σ exp(θⱼ) |
| Parameters | Temperature T | Temperature (inverse) |
| Input | Energies (costs) | Logits (scores) |
| Direction | Lower energy → higher prob | Higher logit → higher prob |
| Physical meaning | Statistical mechanics | ML convention |
| Temperature | Explicit physical parameter | Often implicit (1/τ) |

**When to use Boltzmann**: When you have cost/energy values and want a physically meaningful temperature parameter. The explicit connection to thermodynamics provides intuition (annealing, phase transitions, free energy).

**When to use softmax**: When working within standard ML frameworks where logits are already computed.

They are mathematically equivalent: `softmax(logits)` = `Boltzmann(-logits, T=1)`.

### Boltzmann vs Epsilon-Greedy

| Aspect | Boltzmann | ε-greedy |
|--------|-----------|----------|
| Exploration | Smooth, graded | Binary (random or greedy) |
| Worst action prob | exp(-ΔE/kT)/Z > 0 | ε/n |
| Temperature control | Continuous dial | ε parameter |
| Mathematical properties | Satisfies detailed balance | No theoretical guarantees |
| Convergence | Proven for proper annealing | Proven for decreasing ε |
| Implementation | Slightly more complex | Very simple |

**When to use Boltzmann**: When you need graded exploration (not all "non-greedy" actions are equally bad) or when mathematical properties matter.

**When to use ε-greedy**: When simplicity is paramount and the action space is small.

### Boltzmann vs Thompson Sampling

| Aspect | Boltzmann | Thompson Sampling |
|--------|-----------|-------------------|
| Input | Deterministic energies | Posterior distributions |
| Uncertainty | Only through temperature | Full Bayesian |
| Exploration | Temperature-driven | Naturally from posterior |
| Convergence | Depends on schedule | Bayesian optimal |
| Computational cost | O(n) per selection | O(n) with posterior sampling |
| Context | Cost-based selection | Reward-based bandits |

**When to use Boltzmann**: When you have deterministic cost functions and want temperature-controlled exploration.

**When to use Thompson sampling**: When you have reward distributions with uncertainty and want Bayesian-optimal exploration.

---

## Glossary

**Boltzmann constant (k)** — The physical constant relating temperature to energy. In this crate, k = 1 (natural units). In SI units, k ≈ 1.381 × 10⁻²³ J/K.

**Boltzmann distribution** — The probability distribution over states of a system at thermal equilibrium: P(s) ∝ exp(-E(s)/kT). Named after Ludwig Boltzmann (1844–1906).

**Detailed balance** — The condition P(i)·W(i→j) = P(j)·W(j→i) ensuring that a Markov chain has the desired distribution as its stationary distribution. Equivalent to microscopic reversibility.

**Energy** — In agent systems, a cost or loss function assigning a scalar value to each state. Lower energy = more desirable. Analogous to potential energy in physics.

**Ensemble** — A collection of agents, each in some state with associated energy. In statistical mechanics, an ensemble is the probability distribution over all possible microstates.

**Entropy (S)** — A measure of uncertainty or disorder. Shannon entropy: S = -Σ P(i)·ln P(i). Maximum for uniform distribution, zero for a delta function.

**Free energy (F)** — A thermodynamic potential that systems minimize at equilibrium. Helmholtz: F = E - TS. Represents the tradeoff between low energy and high entropy.

**Inverse temperature (β)** — β = 1/(kT). Controls the sharpness of the Boltzmann distribution. β → ∞ concentrates on the lowest-energy state; β → 0 gives uniform distribution.

**Partition function (Z)** — The normalization constant Z = Σᵢ exp(-βEᵢ). Encodes all thermodynamic information about the system. Also called the "sum over states" (Zustandssumme in German).

**Temperature (T)** — Controls exploration vs exploitation. High T → uniform sampling (explore). Low T → concentrated on low-energy states (exploit). Units: energy (since k=1).

**Thermal equilibrium** — The state where the system's distribution over states is the Boltzmann distribution. No net energy flow between states.

**Variational free energy** — An upper bound on negative log-evidence used in active inference: F = accuracy + complexity. Agents minimize this to maintain accurate beliefs.

---

## References

1. **Boltzmann, L.** (1877). "Über die Beziehung zwischen dem zweiten Hauptsatze der mechanischen Wärmetheorie und der Wahrscheinlichkeitsrechnung respektive den Sätzen über das Wärmegleichgewicht." *Wiener Berichte*, 76, 373–435. — The original paper establishing the connection between entropy and probability.

2. **Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E.** (1953). "Equation of State Calculations by Fast Computing Machines." *The Journal of Chemical Physics*, 21(6), 1087–1092. — Introduced the Metropolis algorithm, the foundation of Monte Carlo sampling.

3. **Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P.** (1983). "Optimization by Simulated Annealing." *Science*, 220(4598), 671–680. — Applied the annealing metaphor from physics to combinatorial optimization.

4. **Jaynes, E. T.** (1957). "Information Theory and Statistical Mechanics." *Physical Review*, 106(4), 620–630. — Showed that statistical mechanics is an application of information theory; the Boltzmann distribution maximizes entropy subject to energy constraints.

5. **Landau, L. D. & Lifshitz, E. M.** (1980). *Statistical Physics, Part 1* (3rd ed.). Butterworth-Heinemann. — The definitive graduate textbook on equilibrium statistical mechanics.

6. **Friston, K.** (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127–138. — Introduces the free energy principle as a unifying theory of brain function.

7. **Friston, K., Kilner, J., & Harrison, L.** (2006). "A free energy principle for the brain." *Journal of Physiology – Paris*, 100(1–3), 70–87. — Technical exposition of variational free energy and active inference.

8. **Loshchilov, I. & Hutter, F.** (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR 2017*. — Introduced cosine annealing schedules for deep learning optimization.

9. **Geman, S. & Geman, D.** (1984). "Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 6(6), 721–741. — Connected Gibbs distributions to computational sampling (Gibbs sampling).

10. **MacKay, D. J. C.** (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press. — Comprehensive treatment connecting information theory, statistical mechanics, and machine learning.

---

## License

MIT License. See [LICENSE](./LICENSE) for details.
