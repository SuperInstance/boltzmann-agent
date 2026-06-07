//! # boltzmann-agent
//!
//! Boltzmann distribution applied to agent action selection and multi-agent systems.
//!
//! This crate provides tools for applying statistical mechanics concepts—particularly
//! the Boltzmann distribution—to computational agents. It enables temperature-controlled
//! action selection, simulated annealing optimization, detailed balance verification,
//! multi-agent ensembles, and free energy-based active inference.
//!
//! # Modules
//!
//! - [`distribution`] — Boltzmann distribution P(s) = exp(-βE(s)) / Z
//! - [`selection`] — Action selection with temperature annealing schedules
//! - [`annealing`] — Simulated annealing optimization
//! - [`equilibrium`] — Detailed balance and Boltzmann equilibrium
//! - [`multi_agent`] — Multi-agent ensembles with competitive/cooperative modes
//! - [`free_energy`] — Helmholtz, Gibbs, and variational free energy

pub mod annealing;
pub mod distribution;
pub mod equilibrium;
pub mod free_energy;
pub mod multi_agent;
pub mod selection;

mod rng;

pub use rng::Xorshift64;
