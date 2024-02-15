// Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
// This product includes software developed at Datadog (https://www.datadoghq.com/).
// Copyright 2024 Datadog, Inc.

use std::{collections::HashMap, hash::Hash, iter, ops::RangeToInclusive};

use itertools::Itertools;
use plotly::{
    common::Title,
    layout::{Axis, AxisType},
    Layout, Plot, Scatter,
};
use rand::{distributions::Standard, thread_rng, Rng, RngCore};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::{
    acc::{Accumulator, NumCooccurrences},
    algo::Algorithm,
    exp::Experiment,
};

#[derive(Clone, Debug)]
pub(crate) struct IndependenceAcrossRanges {
    ranges: Vec<RangeToInclusive<u64>>,
    input_size_bytes: usize,
}

impl IndependenceAcrossRanges {
    pub(crate) fn new(ranges: Vec<RangeToInclusive<u64>>, input_size_bytes: usize) -> Self {
        assert!(ranges.iter().all_unique());
        Self {
            ranges,
            input_size_bytes,
        }
    }
}
impl Experiment for IndependenceAcrossRanges {
    type Accumulator = NumCooccurrences<u64>;
    type Result = IndependenceAcrossRangesResult;

    fn new_accumulator(&self) -> Self::Accumulator {
        NumCooccurrences::new(
            iter::once(0..=self.ranges[0].end)
                .chain(
                    self.ranges
                        .iter()
                        .tuple_windows()
                        .map(|(&r0, &r1)| r0.end + 1..=r1.end),
                )
                .multi_cartesian_product(),
        )
    }

    #[inline]
    fn run(&self, accumulator: &mut Self::Accumulator, algorithm: &impl Algorithm) {
        let mut bytes = vec![0; self.input_size_bytes];
        loop {
            thread_rng().fill_bytes(&mut bytes);
            let hashes = self
                .ranges
                .iter()
                .map(|&range| algorithm.hash(&bytes, 0, range))
                .collect::<Vec<_>>();
            if hashes.iter().all_unique() {
                accumulator.record(hashes);
                break;
            }
        }
    }

    fn result(&self, accumulator: &Self::Accumulator) -> Self::Result {
        let num_keys = accumulator.num_iterations();
        let p_value =
            chi_squared_mutual_independence_test_p_value(accumulator.counts(), self.ranges.len());
        Self::Result { num_keys, p_value }
    }

    fn plot(&self, results_by_algo: &HashMap<String, Vec<Self::Result>>) -> Plot {
        let mut plot = Plot::new();
        results_by_algo
            .iter()
            .sorted_by_key(|&(algo, _)| algo)
            .map(|(algo, results)| {
                Scatter::new(
                    results.iter().map(|r| r.num_keys).collect(),
                    results.iter().map(|r| r.p_value).collect(),
                )
                .name(algo)
            })
            .for_each(|trace| plot.add_trace(trace));
        let layout = Layout::new()
            .title(Title::new(
                "Independence of the hashes of the same key and distinct ranges",
            ))
            .x_axis(
                Axis::new()
                    .title(Title::new("Number of hashed keys"))
                    .type_(AxisType::Log),
            )
            .y_axis(
                Axis::new()
                    .title(Title::new(
                        "p-value of the chi-squared test of independence",
                    ))
                    .range(vec![0.0, 1.0]),
            );
        plot.set_layout(layout);
        plot
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct IndependenceAcrossRangesResult {
    num_keys: u64,
    p_value: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct IndependenceAcrossSeeds {
    range: RangeToInclusive<u64>,
    seeds: Vec<u64>,
    input_size_bytes: usize,
}

impl IndependenceAcrossSeeds {
    pub(crate) fn new(
        range: RangeToInclusive<u64>,
        num_seeds: usize,
        input_size_bytes: usize,
    ) -> Self {
        Self {
            range,
            seeds: iter::repeat_with(|| {
                thread_rng()
                    .sample_iter(Standard)
                    .take(num_seeds)
                    .collect::<Vec<_>>()
            })
            .find(|seeds| seeds.iter().all_unique())
            .unwrap(),
            input_size_bytes,
        }
    }
}

impl Experiment for IndependenceAcrossSeeds {
    type Accumulator = NumCooccurrences<u64>;
    type Result = IndependenceAcrossSeedsResult;

    fn new_accumulator(&self) -> Self::Accumulator {
        NumCooccurrences::new(
            iter::repeat(0..=self.range.end)
                .take(self.seeds.len())
                .multi_cartesian_product(),
        )
    }

    #[inline]
    fn run(&self, accumulator: &mut Self::Accumulator, algorithm: &impl Algorithm) {
        let mut bytes = vec![0; self.input_size_bytes];
        thread_rng().fill_bytes(&mut bytes);
        let hashes = self
            .seeds
            .iter()
            .map(|&seed| algorithm.hash(&bytes, seed, self.range))
            .collect::<Vec<_>>();
        accumulator.record(hashes)
    }

    fn result(&self, accumulator: &Self::Accumulator) -> Self::Result {
        let num_keys = accumulator.num_iterations();
        let p_value =
            chi_squared_mutual_independence_test_p_value(accumulator.counts(), self.seeds.len());
        Self::Result { num_keys, p_value }
    }

    fn plot(&self, results_by_algo: &HashMap<String, Vec<Self::Result>>) -> Plot {
        let mut plot = Plot::new();
        results_by_algo
            .iter()
            .sorted_by_key(|&(algo, _)| algo)
            .map(|(algo, results)| {
                Scatter::new(
                    results.iter().map(|r| r.num_keys).collect(),
                    results.iter().map(|r| r.p_value).collect(),
                )
                .name(algo)
            })
            .for_each(|trace| plot.add_trace(trace));
        let layout = Layout::new()
            .title(Title::new(
                "Independence of the hashes of the same key and distinct seeds",
            ))
            .x_axis(
                Axis::new()
                    .title(Title::new("Number of hashed keys"))
                    .type_(AxisType::Log),
            )
            .y_axis(
                Axis::new()
                    .title(Title::new(
                        "p-value of the chi-squared test of independence",
                    ))
                    .range(vec![0.0, 1.0]),
            );
        plot.set_layout(layout);
        plot
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct IndependenceAcrossSeedsResult {
    num_keys: u64,
    p_value: f64,
}

fn chi_squared_mutual_independence_test_p_value<H: Eq + Hash>(
    num_cooccurrences: &HashMap<Vec<H>, u64>,
    n: usize,
) -> f64 {
    let (marginal_probabilities, num_samples) = {
        let mut p = iter::repeat_with(HashMap::<_, f64>::new)
            .take(n)
            .collect::<Vec<_>>();
        num_cooccurrences.iter().for_each(|(i, &v)| {
            iter::zip(i, &mut p).for_each(|(i_i, p_i)| *p_i.entry(i_i).or_default() += v as f64);
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
