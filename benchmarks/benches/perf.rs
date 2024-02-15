// Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
// This product includes software developed at Datadog (https://www.datadoghq.com/).
// Copyright 2024 Datadog, Inc.

use std::{hint::black_box, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use fliphash::{fliphash_64, fliphash_xxh3_64};
use fliphash_benchmarks::{
    anchor::{mix_bits, AnchorHash},
    jump::jumphash,
};
use rand::{thread_rng, RngCore};
use xxhash_rust::xxh3;

const RANGE_ENDS: [u64; 4] = [10, 1000, 100000, 10000000];
const ANCHOR_MAX_RANGE_END: u64 = 100000; // It uses too much memory otherwise.
const ANCHOR_PROVISIONING_RATIOS: [f64; 4] = [1.0, 1.1, 2.0, 10.0];

fn hash_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("HashU64");
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_millis(1000));
    group.sample_size(1000);

    let mut rng = thread_rng();

    for range_end in RANGE_ENDS {
        group.bench_with_input(
            BenchmarkId::new("Jump", format!("..={}", range_end)),
            &..=range_end,
            |b, &range| {
                let key = rng.next_u64();
                b.iter(|| jumphash(black_box(key), black_box(..=range.end as u32)))
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Flip", format!("..={}", range_end)),
            &..=range_end,
            |b, &range| {
                let key = rng.next_u64();
                b.iter(|| fliphash_64(black_box(key), black_box(range)))
            },
        );
        if range_end <= ANCHOR_MAX_RANGE_END {
            for provisioning_ratio in ANCHOR_PROVISIONING_RATIOS {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("Anchor({})", provisioning_ratio),
                        format!("..={}", range_end),
                    ),
                    &..=range_end,
                    |b, &range| {
                        let w = range.end + 1;
                        let a = (w as f64 * provisioning_ratio) as u64;
                        let anchor_hash = AnchorHash::new(black_box(a), black_box(w));
                        let key = rng.next_u64();
                        b.iter(|| anchor_hash.get_bucket(black_box(key), mix_bits))
                    },
                );
            }
        }
    }
    group.finish();
}

fn hash_bytes_with_xxh3(c: &mut Criterion) {
    let mut group = c.benchmark_group("HashBytes");
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_millis(1000));
    group.sample_size(1000);

    let mut rng = thread_rng();
    let mut bytes = [0_u8; 128];
    for range_end in RANGE_ENDS {
        group.bench_with_input(
            BenchmarkId::new("XXH3_then_Jump", format!("..={}", range_end)),
            &..=range_end as u32,
            |b, &range| {
                rng.fill_bytes(&mut bytes);
                b.iter(|| jumphash(xxh3::xxh3_64(&black_box(bytes)), black_box(range)))
            },
        );
        group.bench_with_input(
            BenchmarkId::new("XXH3_then_Flip", format!("..={}", range_end)),
            &..=range_end,
            |b, &range| {
                rng.fill_bytes(&mut bytes);
                b.iter(|| fliphash_64(xxh3::xxh3_64(&black_box(bytes)), black_box(range)))
            },
        );
        if range_end <= ANCHOR_MAX_RANGE_END {
            for provisioning_ratio in ANCHOR_PROVISIONING_RATIOS {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("XXH3_then_Anchor({})", provisioning_ratio),
                        format!("..={}", range_end),
                    ),
                    &..=range_end,
                    |b, &range| {
                        let w = range.end + 1;
                        let a = (w as f64 * provisioning_ratio) as u64;
                        let anchor_hash = AnchorHash::new(black_box(a), black_box(w));
                        rng.fill_bytes(&mut bytes);
                        b.iter(|| {
                            anchor_hash.get_bucket(xxh3::xxh3_64(&black_box(bytes)), mix_bits)
                        })
                    },
                );
            }
        }
        group.bench_with_input(
            BenchmarkId::new("XXH3_based_Flip", format!("..={}", range_end)),
            &..=range_end,
            |b, &range| {
                rng.fill_bytes(&mut bytes);
                b.iter(|| fliphash_xxh3_64(&black_box(bytes), black_box(range)))
            },
        );
        if range_end <= ANCHOR_MAX_RANGE_END {
            for provisioning_ratio in ANCHOR_PROVISIONING_RATIOS {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("XXH3_based_Anchor({})", provisioning_ratio),
                        format!("..={}", range_end),
                    ),
                    &..=range_end,
                    |b, &range| {
                        let w = range.end + 1;
                        let a = (w as f64 * provisioning_ratio) as u64;
                        let anchor_hash = AnchorHash::new(black_box(a), black_box(w));
                        rng.fill_bytes(&mut bytes);
                        b.iter(|| {
                            anchor_hash.get_bucket(&black_box(bytes), |key, seed| {
                                xxh3::xxh3_64_with_seed(seed, key)
                            })
                        })
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, hash_u64, hash_bytes_with_xxh3);
criterion_main!(benches);
