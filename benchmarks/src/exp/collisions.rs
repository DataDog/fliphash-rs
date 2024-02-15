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

use crate::{
    acc::{Accumulator, NumOccurrences},
    algo::Algorithm,
    exp::Experiment,
};

#[derive(Clone, Debug)]
pub(crate) struct Collisions {
    range: RangeToInclusive<u64>,
    input_size_bytes: usize,
}

impl Collisions {
    pub(crate) fn new(range: RangeToInclusive<u64>, input_size_bytes: usize) -> Self {
        Self {
            range,
            input_size_bytes,
        }
    }
}

impl Experiment for Collisions {
    type Accumulator = NumOccurrences<u64>;
    type Result = CollisionResult;

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
        let num_collisions = accumulator
            .counts()
            .iter()
            .filter(|&&c| c > 1)
            .map(|&c| c as f64)
            .map(|c| c * (c - 1.0) / 2.0)
            .sum::<f64>();
        let c_hat = num_collisions / (num_keys as f64 * (num_keys as f64 - 1.0) / 2.0);
        let normalized_c_hat = c_hat * accumulator.counts().len() as f64;
        Self::Result {
            num_keys,
            num_collisions,
            c_hat,
            normalized_c_hat,
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
                    results.iter().map(|r| r.normalized_c_hat).collect(),
                )
                .name(algo)
            })
            .for_each(|trace| plot.add_trace(trace));
        let layout = Layout::new()
            .title(Title::new("Hash collisions"))
            .x_axis(
                Axis::new()
                    .title(Title::new("Number of hashed keys"))
                    .type_(AxisType::Log),
            )
            .y_axis(Axis::new().title(Title::new("Normalized C hat")));
        plot.set_layout(layout);
        plot
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct CollisionResult {
    num_keys: u64,
    num_collisions: f64,
    c_hat: f64,
    normalized_c_hat: f64,
}
