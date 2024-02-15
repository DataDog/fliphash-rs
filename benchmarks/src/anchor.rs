// Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
// This product includes software developed at Datadog (https://www.datadoghq.com/).
// Copyright 2024 Datadog, Inc.

use std::vec;

#[allow(non_snake_case)] // Matching paper's variable names
pub struct AnchorHash {
    A: Vec<u64>,
    R: Vec<u64>,
    N: u64,
    K: Vec<u64>,
    L: Vec<u64>,
    W: Vec<u64>,
    a: u64,
}

// https://arxiv.org/abs/1812.09674, Algorithm 3 (minimal-memory implementation)
impl AnchorHash {
    pub fn new(a: u64, w: u64) -> Self {
        assert!(a >= w);
        let mut anchor_hash = AnchorHash {
            A: vec![0; usize::try_from(a).unwrap()],
            R: vec![],
            N: w,
            K: (0..a).collect(),
            L: (0..a).collect(),
            W: (0..a).collect(),
            a,
        };
        for b in (w..a).rev() {
            anchor_hash.R.push(b);
            anchor_hash.A[b as usize] = b;
        }
        anchor_hash
    }

    #[inline]
    pub fn get_bucket<K: Copy>(&self, k: K, hash_fn: impl Fn(u64, K) -> u64) -> u64 {
        let mut b = hash_fn(0, k) % self.a;
        while self.A[b as usize] > 0 {
            let mut h = hash_fn(b, k) % self.A[b as usize];
            while self.A[h as usize] >= self.A[b as usize] {
                h = self.K[h as usize];
            }
            b = h
        }
        b
    }

    pub fn add_bucket(&mut self) -> u64 {
        let b = self.R.pop().unwrap();
        self.A[b as usize] = 0;
        self.L[self.W[self.N as usize] as usize] = self.N;
        self.K[b as usize] = b;
        self.W[self.L[b as usize] as usize] = b;
        self.N += 1;
        b
    }

    pub fn remove_bucket(&mut self, b: u64) {
        self.R.push(b);
        self.N -= 1;
        self.A[b as usize] = self.N;
        self.K[b as usize] = self.W[self.N as usize];
        self.W[self.L[b as usize] as usize] = self.W[self.N as usize];
        self.L[self.W[self.N as usize] as usize] = self.L[b as usize];
    }
}

// Similar to the bit mixing of `fliphash_64`.
#[inline(always)]
pub const fn mix_bits(key: u64, seed: u64) -> u64 {
    let mut k = key ^ seed;
    k = (k ^ (k >> 27)).wrapping_mul(0x3C79AC492BA7B653);
    k = (k ^ (k >> 33)).wrapping_mul(0x1C69B3F74AC4AE35);
    k ^ (k >> 27)
}

#[cfg(test)]
mod tests {
    use std::iter;

    use proptest::prelude::*;
    use rand::thread_rng;

    use super::*;

    const MAX_A: u64 = 100;

    #[test]
    fn range() {
        proptest!(|(a in 1..=MAX_A, w in 1..=MAX_A, k: u64)| {
            prop_assume!(w <= a);
            let anchor_hash = AnchorHash::new(a, w);
            prop_assert!((0..w).contains(&anchor_hash.get_bucket(k, mix_bits)));
        });
    }

    #[test]
    fn monotonicity() {
        proptest!(|(a in 1..=MAX_A, w in 1..=MAX_A, k: u64)| {
            prop_assume!(w < a);
            let mut anchor_hash = AnchorHash::new(a, w);
            let b0 = anchor_hash.get_bucket(k, mix_bits);
            let new_bucket = anchor_hash.add_bucket();
            prop_assert_ne!(new_bucket, b0);
            let b1 = anchor_hash.get_bucket(k, mix_bits);
            prop_assert!(b1 == b0 || b1 == new_bucket);
        });
    }

    #[test]
    fn regularity() {
        const MAX_NUM_DRAWS: u64 = 100_000_000;
        const STEP: u64 = 1_000_000;
        const MAX_DELTA: f64 = 1e-2;
        proptest!(ProptestConfig::with_cases(10), |(a in (1..=MAX_A).no_shrink(), w in (1..=MAX_A).no_shrink())| {
            prop_assume!(w <= a);
            dbg!(w, a);
            let mut thread_rng = thread_rng();
            let anchor_hash = AnchorHash::new(a, w);
            let mut hash_counts = vec![0; w as usize];
            let mut regular = false;
            while !regular && hash_counts.iter().sum::<u64>() < MAX_NUM_DRAWS {
                iter::repeat_with(|| thread_rng.next_u64())
                    .map(|key| anchor_hash.get_bucket(key, mix_bits))
                    .take(STEP as usize)
                    .for_each(|hash| hash_counts[hash as usize] += 1);
                let (min_count, max_count) = (
                    *hash_counts.iter().min().unwrap() as f64,
                    *hash_counts.iter().max().unwrap() as f64,
                );
                dbg!(max_count/min_count);
                regular = max_count <= min_count * (1.0 + MAX_DELTA);
            }
            prop_assert!(regular);
        });
    }
}
