use serde::{Deserialize, Serialize};

/// A simple xorshift64 PRNG for reproducible sampling.
///
/// No external `rand` dependency needed. Suitable for Monte Carlo sampling
/// in agent systems where cryptographic security is not required.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new PRNG with the given seed. Panics if seed is 0.
    pub fn new(seed: u64) -> Self {
        assert_ne!(seed, 0, "Xorshift64 seed must be non-zero");
        Self { state: seed }
    }

    /// Create a PRNG seeded from a default value (useful for deterministic tests).
    pub fn default_seed() -> Self {
        Self::new(0xDEAD_BEEF_CAFE_BABE)
    }

    /// Generate the next pseudo-random u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a pseudo-random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate a pseudo-random usize in [0, n).
    pub fn next_usize(&mut self, n: usize) -> usize {
        ((self.next_f64()) * n as f64) as usize
    }

    /// Generate a standard Normal(0, 1) sample using Box-Muller.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let mag = (-2.0 * u1.ln()).sqrt();
        mag * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reproducibility() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_f64_range() {
        let mut rng = Xorshift64::new(123);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "f64 out of range: {v}");
        }
    }

    #[test]
    fn test_normal_approximately_zero_mean() {
        let mut rng = Xorshift64::new(999);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| rng.next_normal()).sum();
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.1, "normal mean too far from zero: {mean}");
    }

    #[test]
    #[should_panic]
    fn test_zero_seed_panics() {
        Xorshift64::new(0);
    }
}
