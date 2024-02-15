// Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
// This product includes software developed at Datadog (https://www.datadoghq.com/).
// Copyright 2024 Datadog, Inc.

use std::{fmt, ops::RangeToInclusive};

use fliphash::{fliphash_64_with_seed, fliphash_xxh3_128_with_seed, fliphash_xxh3_64_with_seed};

use crate::jump::jumphash;

pub(crate) trait Algorithm: fmt::Display {
    fn hash(&self, key: &[u8], seed: u64, range: RangeToInclusive<u64>) -> u64;
}

#[derive(Clone, Debug)]
pub(crate) struct FlipHash64;
impl fmt::Display for FlipHash64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FlipHash (64 bits)")
    }
}
impl Algorithm for FlipHash64 {
    #[inline]
    fn hash(&self, key: &[u8], seed: u64, range: RangeToInclusive<u64>) -> u64 {
        debug_assert!(key.len() >= 8);
        fliphash_64_with_seed(
            u64::from_ne_bytes(key[..8].try_into().unwrap()),
            seed,
            range,
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FlipHashXXH364;
impl fmt::Display for FlipHashXXH364 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FlipHash (XXH3, 64 bits)")
    }
}
impl Algorithm for FlipHashXXH364 {
    #[inline]
    fn hash(&self, key: &[u8], seed: u64, range: RangeToInclusive<u64>) -> u64 {
        fliphash_xxh3_64_with_seed(key, seed, range)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FlipHashXXH3128;
impl fmt::Display for FlipHashXXH3128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FlipHash (XXH3, 128 bits)")
    }
}
impl Algorithm for FlipHashXXH3128 {
    #[inline]
    fn hash(&self, key: &[u8], seed: u64, range: RangeToInclusive<u64>) -> u64 {
        fliphash_xxh3_128_with_seed(key, seed, ..=range.end.into())
            .try_into()
            .unwrap()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct JumpHash;
impl fmt::Display for JumpHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JumpHash")
    }
}
impl Algorithm for JumpHash {
    #[inline]
    fn hash(&self, key: &[u8], seed: u64, range: RangeToInclusive<u64>) -> u64 {
        debug_assert!(key.len() >= 8);
        jumphash(
            u64::from_ne_bytes(key[..8].try_into().unwrap()) ^ seed,
            ..=u32::try_from(range.end).unwrap(),
        )
        .into()
    }
}
