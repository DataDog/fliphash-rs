// Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
// This product includes software developed at Datadog (https://www.datadoghq.com/).
// Copyright 2024 Datadog, Inc.

use std::ops::RangeToInclusive;

#[inline]
pub fn jumphash(key: u64, range: RangeToInclusive<u32>) -> u32 {
    let mut k = key;
    let (mut b, mut j) = (-1_i64, 0_i64);
    while j <= range.end as i64 {
        b = j;
        k = k.wrapping_mul(2862933555777941757).wrapping_add(1);
        j = ((b + 1) as f64 * (f64::from(1_u32 << 31) / ((k >> 33) + 1) as f64)) as i64;
    }
    b as u32
}
