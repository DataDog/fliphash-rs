// Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
// This product includes software developed at Datadog (https://www.datadoghq.com/).
// Copyright 2024 Datadog, Inc.

use std::{collections::HashMap, ops::RangeToInclusive};

use itertools::Itertools;
use plotly::{
    common::Title,
    layout::{Axis, AxisType},
    Layout, Plot, Scatter,
};
use rand::{thread_rng, RngCore};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::{
    acc::{Accumulator, NumOccurrences},
    algo::Algorithm,
    exp::Experiment,
};

#[derive(Clone, Debug)]
pub(crate) struct Regularity {
    range: RangeToInclusive<u64>,
    input_size_bytes: usize,
}

impl Regularity {
    pub(crate) fn new(range: RangeToInclusive<u64>, input_size_bytes: usize) -> Self {
        Self {
            range,
            input_size_bytes,
        }
    }
}

impl Experiment for Regularity {
    type Accumulator = NumOccurrences<u64>;
    type Result = RegularityResult;

    fn new_accumulator(&self) -> Self::Accumulator {
        NumOccurrences::new(
            usize::try_from(self.range.end)
                .unwrap()
                .checked_add(1)
                .unwrap(),
        )
    }

    #[inline]
    fn run(&self, accumulator: &mut Self::Accumulator, algorithm: &impl Algorithm) {
        let mut bytes = vec![0; self.input_size_bytes];
        thread_rng().fill_bytes(&mut bytes);
        let hash = algorithm.hash(&bytes, 0, self.range);
        accumulator.record(hash);
    }

    fn result(&self, accumulator: &Self::Accumulator) -> Self::Result {
        let num_keys = accumulator.num_iterations();
        let range_len = accumulator.counts().len();
        let l1_distance = accumulator
            .counts()
            .iter()
            .map(|&c| c as f64 / num_keys as f64)
            .map(|p| (p - 1.0 / range_len as f64).abs())
            .sum::<f64>();
        let l2_distance = accumulator
            .counts()
            .iter()
            .map(|&c| c as f64 / num_keys as f64)
            .map(|p| (p - 1.0 / range_len as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let p_value = chi_squared_uniformity_test_p_value(accumulator.counts());
        Self::Result {
            num_keys,
            l1_distance,
            l2_distance,
            p_value,
        }
    }

    fn plot(&self, results_by_algo: &HashMap<String, Vec<Self::Result>>) -> Plot {
        let mut plot = Plot::new();
        results_by_algo
            .iter()
            .sorted_by_key(|&(algo, _)| algo)
            .map(|(algo, results)| {
                Scatter::new(
                    results.iter().map(|r| r.num_keys).collect(),
                    results.iter().map(|r| r.l1_distance).collect(),
                )
                .name(algo)
            })
            .for_each(|trace| plot.add_trace(trace));
        let layout = Layout::new()
            .title(Title::new("Regularity of the hash function"))
            .x_axis(
                Axis::new()
                    .title(Title::new("Number of hashed keys"))
                    .type_(AxisType::Log),
            )
            .y_axis(
                Axis::new()
                    .title(Title::new("L2 distance from the uniform distribution"))
                    .type_(AxisType::Log),
            );
        // .legend(Legend::new());
        plot.set_layout(layout);
        plot
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct RegularityResult {
    num_keys: u64,
    l1_distance: f64,
    l2_distance: f64,
    p_value: f64,
}

fn chi_squared_uniformity_test_p_value(num_occurrences: &[u64]) -> f64 {
    let expected_count = num_occurrences.iter().sum::<u64>() as f64 / num_occurrences.len() as f64;

    let statistic = num_occurrences
        .iter()
        .map(|&o| (o as f64 - expected_count).powi(2) / expected_count)
        .sum::<f64>();

    let degrees_of_freedom = num_occurrences.len() as f64 - 1.0;

    1.0 - ChiSquared::new(degrees_of_freedom).unwrap().cdf(statistic)
}
