// Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
// This product includes software developed at Datadog (https://www.datadoghq.com/).
// Copyright 2024 Datadog, Inc.

//! FlipHash is a consistent range-hashing function that hashes an integer
//! `key` into a value of `..=range_end`, where `range_end` is parameterized.
//!
//! It is:
//!
//! - __regular__ (i.e., uniform, balanced): it distributes the keys evenly over
//!   the hash values of the range,
//! - __monotone__ (i.e., stable): when varying the range, key hashes are not
//!   shuffled across the values that stay within the range, and keys can only
//!   be remapped from a hash value now outside of the range (if the range is
//!   narrowed), or to a hash value previously outside of the range (if the
//!   range is enlarged),
//! - __fast__: it has a low computational cost, and constant-time complexity.
//!
//! # Usage
//!
//! ```
//! use fliphash::fliphash_64;
//!
//! let hash = fliphash_64(10427592028180905159, ..=17);
//!
//! assert!((..=17).contains(&hash));
//! ```
//!
//! # Regularity
//!
//! The following code snippet illustrates the regularity of FlipHash.
//!
//! With a large enough number of distinct keys, the numbers of occurrences of
//! the hash values of `range` are relatively close to one another.
//!
//! ```
//! use fliphash::fliphash_64;
//!
//! let mut hash_counts = [0_u64; 18];
//! // Hash a lot of keys; they could be picked randomly.
//! for key in 0_u64..2_000_000_u64 {
//!     let hash = fliphash_64(key, ..=17);
//!     hash_counts[hash as usize] += 1;
//! }
//!
//! let (min_count, max_count) = (
//!     *hash_counts.iter().min().unwrap() as f64,
//!     *hash_counts.iter().max().unwrap() as f64,
//! );
//! let relative_difference = (max_count - min_count) / min_count;
//! assert!(relative_difference < 0.01);
//! ```
//!
//! # Monotonicity
//!
//! The following code snippet illustrates the monotonicity, i.e., the
//! stability, of FlipHash.
//!
//! Given a key, when making the range larger, either the hash of the key is
//! unchanged or it gets a new value that the previous range does not contain.
//!
//! ```
//! use fliphash::fliphash_64;
//!
//! let key = 10427592028180905159;
//!
//! let mut previous_hash = 0;
//! for range_end in 1..1000 {
//!     let hash = fliphash_64(key, ..=range_end);
//!     assert!(hash == previous_hash || hash == range_end);
//!     previous_hash = hash;
//! }
//! ```
//!
//! # Performance
//!
//! FlipHash has constant average and worst-case time complexity.
//!
//! As a comparison, [Jump Consistent Hash](https://arxiv.org/abs/1406.2294) has a time
//! complexity that is logarithmic in the width of the range.
//!
//! ## Evaluation wall times
//!
//! On an Intel Xeon 8375C CPU.
//!
//! | Range | FlipHash | JumpHash |
//! |-|-|-|
//! | `..=10` | 5.9 ns | 8.2 ns |
//! | `..=1000` | 4.7 ns | 25 ns |
//! | `..=1000000` | 5.5 ns | 45 ns |
//! | `..=1000000000` | 6.4 ns | 69 ns |
#![no_std]
use core::ops::RangeToInclusive;

macro_rules! fliphash {
    ($hash_fn: path, $key: expr, $seed: expr, $range: expr, $max_num_iterations: expr) => {
        match $range.end {
            0 => 0,
            _ => {
                let pow2_mask = !0 >> $range.end.leading_zeros(); // == 2^r - 1
                let hash = $hash_fn($key, $seed, 0, 0);
                match fliphash_pow2!($hash_fn, $key, $seed, hash, pow2_mask) {
                    fliphash_pow2 if fliphash_pow2 <= $range.end => fliphash_pow2,
                    _ => {
                        let mut iteration_index = 1; // i
                        if let Some(draw) = loop {
                            if iteration_index > $max_num_iterations {
                                break None;
                            }
                            let draw = $hash_fn($key, $seed, $range.end.ilog2(), iteration_index)
                                & pow2_mask;
                            if draw <= pow2_mask >> 1 {
                                break None;
                            } else if draw <= $range.end {
                                break Some(draw);
                            }
                            iteration_index += 1;
                        } {
                            draw
                        } else {
                            fliphash_pow2!($hash_fn, $key, $seed, hash, pow2_mask >> 1)
                        }
                    }
                }
            }
        }
    };
}

macro_rules! fliphash_pow2 {
    ($hash_fn: path, $key: expr, $seed: expr, $hash: expr, $pow2_mask: expr) => {
        match $hash & $pow2_mask {
            0 => 0,
            masked_hash => {
                let flipper = $hash_fn($key, $seed, masked_hash.ilog2(), 0)
                    & !0 >> masked_hash.leading_zeros() >> 1;
                masked_hash ^ flipper
            }
        }
    };
}

/// Hashes `key` to a value of `range`, uniformly and with stability.
///
/// # Example
///
/// ```
/// use fliphash::fliphash_64;
///
/// let key = 15960427081186311679;
/// let hash_17 = fliphash_64(key, ..=17);
/// let hash_18 = fliphash_64(key, ..=18);
///
/// assert!(hash_17 <= 17);
/// assert!(hash_18 == hash_17 || hash_18 == 18);
/// ```
#[inline]
pub const fn fliphash_64(key: u64, range: RangeToInclusive<u64>) -> u64 {
    fliphash_64_with_seed(key, 0, range)
}

#[inline]
pub const fn fliphash_64_with_seed(key: u64, seed: u64, range: RangeToInclusive<u64>) -> u64 {
    const MAX_NUM_ITERATIONS: u32 = 64;
    #[inline(always)]
    const fn hash(key: u64, seed: u64, bit_len: u32, iteration_index: u32) -> u64 {
        // Inspired by https://mostlymangling.blogspot.com/2019/12/stronger-better-morer-moremur-better.html
        let mut k = key ^ seed;
        k = k.wrapping_mul(bit_len as u64 * 2 + 1);
        k = (k ^ (k >> 27)).wrapping_mul(0x3C79AC492BA7B653);
        k = k.wrapping_mul(iteration_index as u64 * 2 + 1);
        k = (k ^ (k >> 33)).wrapping_mul(0x1C69B3F74AC4AE35);
        k ^ (k >> 27)
    }
    fliphash!(hash, key, seed, range, MAX_NUM_ITERATIONS)
}

#[cfg(feature = "xxh3")]
const XXH3_MAX_NUM_ITERATIONS: u32 = 64;

#[cfg(feature = "xxh3")]
#[inline]
pub fn fliphash_xxh3_64(key: &[u8], range: RangeToInclusive<u64>) -> u64 {
    fliphash_xxh3_64_with_seed(key, 0, range)
}

#[cfg(feature = "xxh3")]
#[inline]
pub fn fliphash_xxh3_64_with_seed(key: &[u8], seed: u64, range: RangeToInclusive<u64>) -> u64 {
    #[inline(always)]
    fn hash(key: &[u8], seed: u64, bit_len: u32, iteration_index: u32) -> u64 {
        xxhash_rust::xxh3::xxh3_64_with_seed(
            key,
            seed ^ (bit_len as u64 + ((iteration_index as u64) << 32)),
        )
    }
    fliphash!(hash, key, seed, range, XXH3_MAX_NUM_ITERATIONS)
}

#[cfg(feature = "xxh3")]
#[inline]
pub const fn fliphash_const_xxh3_64(key: &[u8], range: RangeToInclusive<u64>) -> u64 {
    fliphash_const_xxh3_64_with_seed(key, 0, range)
}

#[cfg(feature = "xxh3")]
#[inline]
pub const fn fliphash_const_xxh3_64_with_seed(
    key: &[u8],
    seed: u64,
    range: RangeToInclusive<u64>,
) -> u64 {
    #[inline(always)]
    const fn hash(key: &[u8], seed: u64, bit_len: u32, iteration_index: u32) -> u64 {
        xxhash_rust::const_xxh3::xxh3_64_with_seed(
            key,
            seed ^ (bit_len as u64 + ((iteration_index as u64) << 32)),
        )
    }
    fliphash!(hash, key, seed, range, XXH3_MAX_NUM_ITERATIONS)
}

#[cfg(feature = "xxh3")]
#[inline]
pub fn fliphash_xxh3_128(key: &[u8], range: RangeToInclusive<u128>) -> u128 {
    fliphash_xxh3_128_with_seed(key, 0, range)
}

#[cfg(feature = "xxh3")]
#[inline]
pub fn fliphash_xxh3_128_with_seed(key: &[u8], seed: u64, range: RangeToInclusive<u128>) -> u128 {
    #[inline(always)]
    fn hash(key: &[u8], seed: u64, bit_len: u32, iteration_index: u32) -> u128 {
        xxhash_rust::xxh3::xxh3_128_with_seed(
            key,
            seed ^ (bit_len as u64 + ((iteration_index as u64) << 32)),
        )
    }
    fliphash!(hash, key, seed, range, XXH3_MAX_NUM_ITERATIONS)
}

#[cfg(feature = "xxh3")]
#[inline]
pub const fn fliphash_const_xxh3_128(key: &[u8], range: RangeToInclusive<u128>) -> u128 {
    fliphash_const_xxh3_128_with_seed(key, 0, range)
}

#[cfg(feature = "xxh3")]
#[inline]
pub const fn fliphash_const_xxh3_128_with_seed(
    key: &[u8],
    seed: u64,
    range: RangeToInclusive<u128>,
) -> u128 {
    #[inline(always)]
    const fn hash(key: &[u8], seed: u64, bit_len: u32, iteration_index: u32) -> u128 {
        xxhash_rust::const_xxh3::xxh3_128_with_seed(
            key,
            seed ^ (bit_len as u64 + ((iteration_index as u64) << 32)),
        )
    }
    fliphash!(hash, key, seed, range, XXH3_MAX_NUM_ITERATIONS)
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    extern crate std;

    use alloc::vec::Vec;
    use core::{array, fmt::Debug};
    use std::{
        collections::HashMap,
        format,
        hash::Hash,
        iter,
        ops::{Range, RangeToInclusive},
        println, vec,
    };

    use itertools::Itertools;
    use num_traits::{one, zero, NumCast, PrimInt};
    use ordered_float::NotNan;
    use proptest::prelude::*;
    use proptest_derive::Arbitrary;
    use rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, seq::IteratorRandom,
        thread_rng, SeedableRng,
    };
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    use test_case::test_case;

    // Homogeneize types and customize input generation.
    macro_rules! wrap {
        ($fliphash_fn: ident, $fliphash_with_seed_fn: ident, $key_ty: ty, $seed_ty: ty, $hash_ty: ty) => {
            fn $fliphash_fn(key: &$key_ty, range: RangeToInclusive<$hash_ty>) -> $hash_ty {
                super::$fliphash_fn(key.into(), range)
            }
            fn $fliphash_with_seed_fn(
                key: &$key_ty,
                seed: $seed_ty,
                range: RangeToInclusive<$hash_ty>,
            ) -> $hash_ty {
                super::$fliphash_with_seed_fn(key.into(), seed, range)
            }
        };
    }
    wrap!(fliphash_64, fliphash_64_with_seed, U64Key, u64, u64);
    #[cfg(feature = "xxh3")]
    wrap!(
        fliphash_xxh3_64,
        fliphash_xxh3_64_with_seed,
        Bytes,
        u64,
        u64
    );
    #[cfg(feature = "xxh3")]
    wrap!(
        fliphash_const_xxh3_64,
        fliphash_const_xxh3_64_with_seed,
        Bytes,
        u64,
        u64
    );
    #[cfg(feature = "xxh3")]
    wrap!(
        fliphash_xxh3_128,
        fliphash_xxh3_128_with_seed,
        Bytes,
        u64,
        u128
    );
    #[cfg(feature = "xxh3")]
    wrap!(
        fliphash_const_xxh3_128,
        fliphash_const_xxh3_128_with_seed,
        Bytes,
        u64,
        u128
    );

    #[derive(Arbitrary, Debug)]
    struct U64Key(u64);
    impl Distribution<U64Key> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> U64Key {
            U64Key(self.sample(rng))
        }
    }
    impl From<&U64Key> for u64 {
        fn from(value: &U64Key) -> Self {
            value.0
        }
    }

    #[derive(Arbitrary, Debug)]
    struct Bytes(Vec<u8>);
    impl Distribution<Bytes> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Bytes {
            // Small byte slices, if taken in same proportions as other slice
            // lengths, bias the distribution of XXH3 hashes.
            const SIZE_RANGE: Range<usize> = 10..100;
            let mut bytes = vec![0; rng.gen_range(SIZE_RANGE)];
            rng.fill_bytes(&mut bytes);
            Bytes(bytes)
        }
    }
    impl<'a> From<&'a Bytes> for &'a [u8] {
        fn from(value: &'a Bytes) -> Self {
            value.0.as_slice()
        }
    }

    fn mostly_small_ranges<H>() -> impl Strategy<Value = RangeToInclusive<H>>
    where
        H: NumCast + Arbitrary,
    {
        prop_oneof! {
            // Favor small ranges.
            80 => (..200_u64).prop_map(NumCast::from).prop_map(Option::unwrap),
            20 => any::<H>()
        }
        .prop_map(|range_end| ..=range_end)
    }

    #[test_case(fliphash_64, fliphash_64_with_seed)]
    #[cfg_attr(
        feature = "xxh3",
        test_case(fliphash_xxh3_64, fliphash_xxh3_64_with_seed),
        test_case(fliphash_const_xxh3_64, fliphash_const_xxh3_64_with_seed),
        test_case(fliphash_xxh3_128, fliphash_xxh3_128_with_seed),
        test_case(fliphash_const_xxh3_128, fliphash_const_xxh3_128_with_seed)
    )]
    fn default_seed<K, S, H>(
        fliphash: impl Fn(&K, RangeToInclusive<H>) -> H,
        fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H,
    ) where
        K: Arbitrary,
        S: Default,
        H: PrimInt + Arbitrary,
    {
        proptest!(|(key: K, range in mostly_small_ranges())| {
            prop_assert_eq!(fliphash(&key, range), fliphash_with_seed(&key, Default::default(), range))
        });
    }

    #[test_case(fliphash_64_with_seed)]
    #[cfg_attr(
        feature = "xxh3",
        test_case(fliphash_xxh3_64_with_seed),
        test_case(fliphash_const_xxh3_64_with_seed),
        test_case(fliphash_xxh3_128_with_seed),
        test_case(fliphash_const_xxh3_128_with_seed)
    )]
    fn zero_range_input<K, S, H>(fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H)
    where
        K: Arbitrary,
        S: Arbitrary,
        H: PrimInt + Debug,
    {
        proptest!(|(key: K, seed: S)| {
            prop_assert_eq!(fliphash_with_seed(&key, seed, ..=zero()), zero())
        });
    }

    #[test_case(fliphash_64_with_seed)]
    #[cfg_attr(
        feature = "xxh3",
        test_case(fliphash_xxh3_64_with_seed),
        test_case(fliphash_const_xxh3_64_with_seed),
        test_case(fliphash_xxh3_128_with_seed),
        test_case(fliphash_const_xxh3_128_with_seed)
    )]

    fn full_range_input<K, S, H>(fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H)
    where
        K: Arbitrary,
        S: Arbitrary,
        H: PrimInt,
    {
        proptest!(|(key: K, seed: S)| {
            fliphash_with_seed(&key, seed, ..=H::max_value());
        });
    }

    #[test_case(fliphash_64_with_seed)]
    #[cfg_attr(
        feature = "xxh3",
        test_case(fliphash_xxh3_64_with_seed),
        test_case(fliphash_const_xxh3_64_with_seed),
        test_case(fliphash_xxh3_128_with_seed),
        test_case(fliphash_const_xxh3_128_with_seed)
    )]
    fn in_range<K, S, H>(fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H)
    where
        K: Arbitrary,
        S: Arbitrary + 'static,
        H: PrimInt + Arbitrary + 'static,
    {
        proptest!(|(key: K, seed: S, range in mostly_small_ranges())| {
            prop_assert!(range.contains(&fliphash_with_seed(&key, seed, range)))
        });
    }

    #[test_case(fliphash_64_with_seed)]
    #[cfg_attr(
        feature = "xxh3",
        test_case(fliphash_xxh3_64_with_seed),
        test_case(fliphash_const_xxh3_64_with_seed),
        test_case(fliphash_xxh3_128_with_seed),
        test_case(fliphash_const_xxh3_128_with_seed)
    )]
    fn monotonicity<K, S, H>(fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H)
    where
        K: Arbitrary,
        S: Copy + Arbitrary + 'static,
        H: PrimInt + Arbitrary + 'static,
    {
        proptest!(ProptestConfig::with_cases(100000), |(key: K, seed: S, range1 in mostly_small_ranges(), range2 in mostly_small_ranges())| {
            let (smaller_range, larger_range) = if range1.end < range2.end {
                (range1, range2)
            } else {
                (range2, range1)
            };
            let smaller_range_hash = fliphash_with_seed(&key, seed, smaller_range);
            let larger_range_hash = fliphash_with_seed(&key, seed, larger_range);
            prop_assert!(smaller_range_hash == larger_range_hash || !smaller_range.contains(&larger_range_hash));
        });
    }

    #[test_case(fliphash_64_with_seed)]
    #[cfg_attr(
        feature = "xxh3",
        test_case(fliphash_xxh3_64_with_seed),
        test_case(fliphash_const_xxh3_64_with_seed),
        test_case(fliphash_xxh3_128_with_seed),
        test_case(fliphash_const_xxh3_128_with_seed)
    )]
    fn regularity<K, S, H>(fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H)
    where
        S: Copy + Debug,
        H: PrimInt + Debug,
        Standard: Distribution<K> + Distribution<S>,
    {
        #[derive(Debug)]
        struct TestCase<S, H> {
            seed: S,
            range: RangeToInclusive<H>,
            num_hashes: usize,
            key_rng_seed: u64,
        }

        let mut rng = thread_rng();
        assert_statistical_hypothesis(
            iter::repeat_with(|| TestCase {
                seed: rng.gen::<S>(),
                range: ..=H::from(rng.gen_range(2..200)).unwrap(),
                num_hashes: rng.gen_range(100..10000),
                key_rng_seed: rng.next_u64(),
            }),
            |test_case| {
                let mut num_occurrences = vec![0_u64; test_case.range.end.to_usize().unwrap() + 1];
                StdRng::seed_from_u64(test_case.key_rng_seed)
                    .sample_iter(Standard)
                    .map(|key| fliphash_with_seed(&key, test_case.seed, test_case.range))
                    .take(test_case.num_hashes)
                    .for_each(|hash| num_occurrences[hash.to_usize().unwrap()] += 1);
                chi_squared_uniformity_test_p_value(&num_occurrences)
            },
        );
    }

    #[test_case(fliphash_64_with_seed)]
    #[cfg_attr(
        feature = "xxh3",
        test_case(fliphash_xxh3_64_with_seed),
        test_case(fliphash_const_xxh3_64_with_seed),
        test_case(fliphash_xxh3_128_with_seed),
        test_case(fliphash_const_xxh3_128_with_seed)
    )]
    fn pairwise_independence_across_seeds<K, S, H>(
        fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H,
    ) where
        S: Copy + Eq + Hash + Debug,
        H: PrimInt + Hash + Debug,
        Standard: Distribution<K> + Distribution<S>,
    {
        n_wise_independence_across_seeds::<2, _, _, _>(fliphash_with_seed);
    }

    #[test_case(fliphash_64_with_seed)]
    #[cfg_attr(
        feature = "xxh3",
        test_case(fliphash_xxh3_64_with_seed),
        test_case(fliphash_const_xxh3_64_with_seed),
        test_case(fliphash_xxh3_128_with_seed),
        test_case(fliphash_const_xxh3_128_with_seed)
    )]
    fn three_wise_independence_across_seeds<K, S, H>(
        fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H,
    ) where
        S: Copy + Eq + Hash + Debug,
        H: PrimInt + Hash + Debug,
        Standard: Distribution<K> + Distribution<S>,
    {
        n_wise_independence_across_seeds::<3, _, _, _>(fliphash_with_seed);
    }

    fn n_wise_independence_across_seeds<const N: usize, K, S, H>(
        fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H,
    ) where
        S: Copy + Eq + Hash + Debug,
        H: PrimInt + Hash + Debug,
        Standard: Distribution<K> + Distribution<S>,
    {
        #[derive(Debug)]
        struct TestCase<S, H, const N: usize> {
            seeds: [S; N],
            range: RangeToInclusive<H>,
            num_hashes: usize,
            key_rng_seed: u64,
        }

        let mut rng = thread_rng();
        assert_statistical_hypothesis(
            iter::repeat_with(|| TestCase {
                seeds: array::from_fn::<S, N, _>(|_| rng.gen()),
                range: ..=NumCast::from(rng.gen_range(5..20)).unwrap(),
                num_hashes: rng.gen_range(100..10000),
                key_rng_seed: rng.next_u64(),
            })
            .filter(|test_case| test_case.seeds.iter().all_unique()),
            |test_case| {
                // ranges = [..=1, ..=1] => [[0, 0], [0, 1], [1, 0], [1, 1]
                let mut num_cooccurrences =
                    iter::repeat(range_inclusive(zero(), test_case.range.end))
                        .take(N)
                        .multi_cartesian_product()
                        .map(<[H; N]>::try_from)
                        .map(Result::unwrap)
                        .map(|hashes| (hashes, 0_u64))
                        .collect::<HashMap<_, _>>();
                StdRng::seed_from_u64(test_case.key_rng_seed)
                    .sample_iter(Standard)
                    .map(|key| {
                        test_case
                            .seeds
                            .map(|seed| fliphash_with_seed(&key, seed, test_case.range))
                    })
                    .take(test_case.num_hashes)
                    .for_each(|hashes| *num_cooccurrences.get_mut(&hashes).unwrap() += 1);
                chi_squared_mutual_independence_test_p_value(&num_cooccurrences)
            },
        );
    }

    #[test_case(fliphash_64_with_seed)]
    #[cfg_attr(
        feature = "xxh3",
        test_case(fliphash_xxh3_64_with_seed),
        test_case(fliphash_const_xxh3_64_with_seed),
        test_case(fliphash_xxh3_128_with_seed),
        test_case(fliphash_const_xxh3_128_with_seed)
    )]
    fn pairwise_independence_given_distinct_hashes<K, S, H>(
        fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H,
    ) where
        S: Copy + Debug,
        H: PrimInt + Hash + Debug,
        Standard: Distribution<K> + Distribution<S>,
    {
        n_wise_independence_given_distinct_hashes::<2, _, _, _>(fliphash_with_seed);
    }

    #[test_case(fliphash_64_with_seed)]
    #[cfg_attr(
        feature = "xxh3",
        test_case(fliphash_xxh3_64_with_seed),
        test_case(fliphash_const_xxh3_64_with_seed),
        test_case(fliphash_xxh3_128_with_seed),
        test_case(fliphash_const_xxh3_128_with_seed)
    )]
    fn three_wise_independence_given_distinct_hashes<K, S, H>(
        fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H,
    ) where
        S: Copy + Debug,
        H: PrimInt + Hash + Debug,
        Standard: Distribution<K> + Distribution<S>,
    {
        n_wise_independence_given_distinct_hashes::<3, _, _, _>(fliphash_with_seed);
    }

    fn n_wise_independence_given_distinct_hashes<const N: usize, K, S, H>(
        fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H,
    ) where
        S: Copy + Debug,
        H: PrimInt + Hash + Debug,
        Standard: Distribution<K> + Distribution<S>,
    {
        #[derive(Debug)]
        struct TestCase<S, H, const N: usize> {
            seed: S,
            ranges: [RangeToInclusive<H>; N],
            num_hashes: usize,
            key_rng_seed: u64,
        }

        let mut rng = thread_rng();
        assert_statistical_hypothesis(
            iter::repeat_with(|| TestCase {
                seed: rng.gen(),
                ranges: (1_u128..20)
                    .choose_multiple(&mut rng, N)
                    .iter()
                    .sorted()
                    .map(|&range_end| ..=NumCast::from(range_end).unwrap())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                num_hashes: rng.gen_range(100..10000),
                key_rng_seed: rng.next_u64(),
            })
            // Filter out test cases with no degrees of freedom along at least one variable.
            .filter(|test_case| {
                test_case
                    .ranges
                    .iter()
                    .tuple_windows()
                    .all(|(r0, r1)| r1.end > r0.end + one())
            }),
            |test_case| {
                // ranges = [..=1, ..=4] => [[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]]
                let mut num_cooccurrences =
                    iter::once(range_inclusive(zero(), test_case.ranges[0].end))
                        .chain(
                            test_case
                                .ranges
                                .iter()
                                .tuple_windows()
                                .map(|(r0, r1)| range_inclusive(r0.end + one(), r1.end)),
                        )
                        .multi_cartesian_product()
                        .map(<[H; N]>::try_from)
                        .map(Result::unwrap)
                        .map(|hashes| (hashes, 0_u64))
                        .collect::<HashMap<_, _>>();
                StdRng::seed_from_u64(test_case.key_rng_seed)
                    .sample_iter(Standard)
                    .map(|key| {
                        test_case
                            .ranges
                            .map(|range| fliphash_with_seed(&key, test_case.seed, range))
                    })
                    .filter(|hashes| hashes.iter().all_unique())
                    .take(test_case.num_hashes)
                    .for_each(|hashes| *num_cooccurrences.get_mut(&hashes).unwrap() += 1);
                chi_squared_mutual_independence_test_p_value(&num_cooccurrences)
            },
        );
    }

    fn assert_statistical_hypothesis<C: Debug>(
        test_cases: impl IntoIterator<Item = C>,
        p_value: impl Fn(&C) -> f64,
    ) {
        const SIGNIFICANCE_LEVEL: f64 = 0.05;
        const NUM_TESTS: usize = 50;
        const MAX_NUM_REJECTIONS: usize = 10;

        let p_values = test_cases
            .into_iter()
            .take(NUM_TESTS)
            .map(|test_case| (p_value(&test_case), test_case))
            .sorted_by_key(|&(p_value, _)| NotNan::new(p_value).unwrap())
            .collect::<Vec<_>>();

        let num_rejections = p_values
            .iter()
            .filter(|&(p_value, _)| p_value < &SIGNIFICANCE_LEVEL)
            .count();

        let output_str = format!(
            "{}/{} null hypothesis rejections at significance level {}; p-values: {}",
            num_rejections,
            NUM_TESTS,
            SIGNIFICANCE_LEVEL,
            [0.25, 0.5, 0.75]
                .into_iter()
                .map(|q| {
                    format!(
                        "p{:.0}: {:.3} ",
                        q * 100.0,
                        p_values[(q * p_values.len() as f64) as usize].0
                    )
                })
                .join(", ")
        );
        if num_rejections > MAX_NUM_REJECTIONS {
            panic!("{}", output_str);
        } else {
            println!("{}", output_str);
        }
    }

    fn chi_squared_uniformity_test_p_value(num_occurrences: &Vec<u64>) -> f64 {
        let expected_count =
            num_occurrences.iter().sum::<u64>() as f64 / num_occurrences.len() as f64;

        let statistic = num_occurrences
            .iter()
            .map(|&o| (o as f64 - expected_count).powi(2) / expected_count)
            .sum::<f64>();

        let degrees_of_freedom = num_occurrences.len() as f64 - 1.0;

        1.0 - ChiSquared::new(degrees_of_freedom).unwrap().cdf(statistic)
    }

    fn chi_squared_mutual_independence_test_p_value<H: Eq + Hash, const N: usize>(
        num_cooccurrences: &HashMap<[H; N], u64>,
    ) -> f64 {
        let (marginal_probabilities, num_samples) = {
            let mut p = iter::repeat_with(HashMap::<_, f64>::new)
                .take(N)
                .collect::<Vec<_>>();
            num_cooccurrences.iter().for_each(|(i, &v)| {
                iter::zip(i, &mut p)
                    .for_each(|(i_i, p_i)| *p_i.entry(i_i).or_default() += v as f64);
            });
            let n = p[0].values().sum::<f64>();
            p.iter_mut()
                .flat_map(|p_i| p_i.values_mut())
                .for_each(|p| *p /= n);
            p.iter()
                .for_each(|p_i| assert!((p_i.values().sum::<f64>() - 1.0).abs() < 1e-2));
            (p, n)
        };

        let statistic = num_cooccurrences
            .iter()
            .map(|(i, &o)| {
                let joint_probability = iter::zip(&marginal_probabilities, i)
                    .map(|(p_i, i_i)| *p_i.get(&i_i).unwrap())
                    .product::<f64>();
                let e = joint_probability * num_samples;
                (o as f64 - e).powi(2) / e
            })
            .sum::<f64>();

        let degrees_of_freedom = (marginal_probabilities
            .iter()
            .map(HashMap::len)
            .product::<usize>()
            - 1)
            - (marginal_probabilities
                .iter()
                .map(HashMap::len)
                .map(|len| len - 1)
                .sum::<usize>());

        assert!(degrees_of_freedom > 0);
        1.0 - ChiSquared::new(degrees_of_freedom as f64)
            .unwrap()
            .cdf(statistic)
    }

    #[cfg(feature = "xxh3")]
    #[cfg_attr(
        feature = "xxh3",
        test_case(fliphash_xxh3_64_with_seed, fliphash_const_xxh3_64_with_seed),
        test_case(fliphash_xxh3_128_with_seed, fliphash_const_xxh3_128_with_seed)
    )]
    fn const_variant_compatibility<K, S, H>(
        fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H,
        const_fliphash_with_seed: impl Fn(&K, S, RangeToInclusive<H>) -> H,
    ) where
        K: Arbitrary,
        S: Copy + Arbitrary + 'static,
        H: PrimInt + Arbitrary + 'static,
    {
        proptest!(|(key: K, seed: S, range in mostly_small_ranges())| {
            prop_assert_eq!(fliphash_with_seed(&key, seed, range), const_fliphash_with_seed(&key, seed, range));
        });
    }

    // To avoid the use of unsafe Step.
    fn range_inclusive<H: PrimInt>(start: H, end: H) -> impl Iterator<Item = H> + Clone {
        iter::successors(Some(start), |&h| Some(h + one())).take_while(move |&h| h <= end)
    }
}
