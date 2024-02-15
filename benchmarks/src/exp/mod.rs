// Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
// This product includes software developed at Datadog (https://www.datadoghq.com/).
// Copyright 2024 Datadog, Inc.

use std::collections::HashMap;

use plotly::Plot;

use crate::{acc::Accumulator, algo::Algorithm};

mod collisions;
mod independence;
mod regularity;
pub(crate) use collisions::Collisions;
pub(crate) use independence::{IndependenceAcrossRanges, IndependenceAcrossSeeds};
pub(crate) use regularity::Regularity;

pub(crate) trait Experiment {
    type Accumulator: Accumulator;
    type Result;

    fn new_accumulator(&self) -> Self::Accumulator;

    fn run(&self, accumulator: &mut Self::Accumulator, algorithm: &impl Algorithm);

    fn accumulate(&self, algorithm: &impl Algorithm, num_iterations: u64) -> Self::Accumulator {
        let mut accumulator = self.new_accumulator();
        for _ in 0..num_iterations {
            self.run(&mut accumulator, algorithm);
        }
        accumulator
    }

    fn result(&self, accumulator: &Self::Accumulator) -> Self::Result;

    fn plot(&self, results_by_algo: &HashMap<std::string::String, Vec<Self::Result>>) -> Plot;
}
