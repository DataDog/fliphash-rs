// Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
// This product includes software developed at Datadog (https://www.datadoghq.com/).
// Copyright 2024 Datadog, Inc.

use std::{
    collections::HashMap,
    fmt,
    fs::{create_dir_all, File},
    io::Write,
    path::Path,
    sync::mpsc,
    thread,
};

mod acc;
mod algo;
mod exp;
mod jump;

use acc::Accumulator;
use algo::{FlipHash64, FlipHashXXH3128, FlipHashXXH364, JumpHash};
use clap::Parser;
use exp::{Collisions, Experiment, IndependenceAcrossRanges, IndependenceAcrossSeeds, Regularity};
use itertools::Itertools;
use plotly::ImageFormat;
use serde::{Deserialize, Serialize};

const RESULT_DIR: &str = "results";
const DEFAULT_ALGORITHMS: [Algorithm; 4] = [
    Algorithm::FlipHash64,
    Algorithm::FlipHashXXH364,
    Algorithm::FlipHashXXH3128,
    Algorithm::JumpHash,
];

#[derive(Parser, Debug)]
enum Command {
    /// Tests the uniformity of the distribution of hashes using a chi-squared
    /// test.
    Regularity {
        #[clap(short, long)]
        num_resources: u64,
        #[clap(short, long)]
        input_size_bytes: usize,
        #[clap(short, long, default_values_t=DEFAULT_ALGORITHMS)]
        algorithms: Vec<Algorithm>,
    },

    /// Compares the number of collisions with the expected value if the
    /// distribution is uniform. The number of collisions is related to the L2
    /// distance to the uniform distribution, so this is another way to test for
    /// regularity.
    Collisions {
        #[clap(short, long)]
        num_resources: u64,
        #[clap(short, long)]
        input_size_bytes: usize,
        #[clap(short, long, default_values_t=DEFAULT_ALGORITHMS)]
        algorithms: Vec<Algorithm>,
    },

    /// Tests the mutual independence across a given number of ranges, given
    /// that hashes are pairwise distinct, using a chi-squared test.
    IndependenceAcrossRanges {
        #[clap(short, long)]
        num_resources: Vec<u64>,
        #[clap(short, long)]
        input_size_bytes: usize,
        #[clap(short, long, default_values_t=DEFAULT_ALGORITHMS)]
        algorithms: Vec<Algorithm>,
    },

    /// Tests the mutual independence acros seeds using a chi-squared test.
    IndependenceAcrossSeeds {
        #[clap(short, long)]
        num_resources: u64,
        #[clap(short, long)]
        num_seeds: usize,
        #[clap(short, long)]
        input_size_bytes: usize,
        #[clap(short, long, default_values_t=DEFAULT_ALGORITHMS)]
        algorithms: Vec<Algorithm>,
    },
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum Algorithm {
    #[clap(name = "fliphash-64")]
    FlipHash64,
    #[clap(name = "fliphash-xxh3-64")]
    FlipHashXXH364,
    #[clap(name = "fliphash-xxh3-128")]
    FlipHashXXH3128,
    #[clap(name = "jumphash")]
    JumpHash,
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Algorithm::FlipHash64 => "fliphash-64",
                Algorithm::FlipHashXXH364 => "fliphash-xxh3-64",
                Algorithm::FlipHashXXH3128 => "fliphash-xxh3-128",
                Algorithm::JumpHash => "jumphash",
            }
        )
    }
}

fn main() {
    match Command::parse() {
        Command::Regularity {
            num_resources,
            input_size_bytes,
            algorithms,
        } => {
            let output_dir = format!("{RESULT_DIR}/regularity");
            create_dir_all(&output_dir).unwrap();
            let output_path =
                format!("{output_dir}/{input_size_bytes}_bytes_{num_resources}_resources");
            run_experiment(
                &mut File::create(format!("{output_path}.jsonl")).unwrap(),
                format!("{output_path}.svg"),
                Regularity::new(..=num_resources - 1, input_size_bytes),
                algorithms,
            );
        }
        Command::Collisions {
            num_resources,
            input_size_bytes,
            algorithms,
        } => {
            let output_dir = format!("{RESULT_DIR}/collisions");
            create_dir_all(&output_dir).unwrap();
            let output_path =
                format!("{output_dir}/{input_size_bytes}_bytes_{num_resources}_resources");
            run_experiment(
                &mut File::create(format!("{output_path}.jsonl")).unwrap(),
                format!("{output_path}.svg"),
                Collisions::new(..=num_resources - 1, input_size_bytes),
                algorithms,
            );
        }
        Command::IndependenceAcrossRanges {
            num_resources,
            input_size_bytes,
            algorithms,
        } => {
            let output_dir = format!("{RESULT_DIR}/independence_across_ranges");
            create_dir_all(&output_dir).unwrap();
            let output_path = format!(
                "{output_dir}/{input_size_bytes}_bytes_{}_resources",
                num_resources.iter().join("_")
            );
            run_experiment(
                &mut File::create(format!("{output_path}.jsonl")).unwrap(),
                format!("{output_path}.svg"),
                IndependenceAcrossRanges::new(
                    num_resources
                        .iter()
                        .map(|n| n - 1)
                        .sorted()
                        .map(|end| ..=end)
                        .collect::<Vec<_>>(),
                    input_size_bytes,
                ),
                algorithms,
            )
        }
        Command::IndependenceAcrossSeeds {
            num_resources,
            num_seeds,
            input_size_bytes,
            algorithms,
        } => {
            let output_dir = format!("{RESULT_DIR}/independence_across_seeds");
            create_dir_all(&output_dir).unwrap();
            let output_path = format!(
                "{output_dir}/{input_size_bytes}_bytes_{num_seeds}_seeds_{num_resources}_resources"
            );
            run_experiment(
                &mut File::create(format!("{output_path}.jsonl")).unwrap(),
                format!("{output_path}.svg"),
                IndependenceAcrossSeeds::new(..=num_resources - 1, num_seeds, input_size_bytes),
                algorithms,
            )
        }
    }
}

fn run_experiment<E>(
    output: &mut impl Write,
    svg_path: impl AsRef<Path>,
    experiment: E,
    algorithms: Vec<Algorithm>,
) where
    E: Experiment + Clone + Send + 'static,
    <E as Experiment>::Accumulator: Send,
    <E as Experiment>::Result: Serialize,
{
    const STEP_SIZE: u64 = 10_000_000;
    const PLOT_EVERY: u64 = 1_000_000_000;

    assert!(!algorithms.is_empty());

    let (tx, rx) = mpsc::channel();
    for _ in 0..usize::from(thread::available_parallelism().unwrap()) - 1 {
        let thread_tx = tx.clone();
        let thread_experiment = experiment.clone();
        let thread_algorithms = algorithms.clone();
        thread::spawn(move || loop {
            for algorithm in &thread_algorithms {
                match algorithm {
                    Algorithm::FlipHash64 => {
                        thread_tx
                            .send((
                                format!("{}", FlipHash64),
                                thread_experiment.accumulate(&FlipHash64, STEP_SIZE),
                            ))
                            .unwrap();
                    }
                    Algorithm::FlipHashXXH364 => {
                        thread_tx
                            .send((
                                format!("{}", FlipHashXXH364),
                                thread_experiment.accumulate(&FlipHashXXH364, STEP_SIZE),
                            ))
                            .unwrap();
                    }
                    Algorithm::FlipHashXXH3128 => {
                        thread_tx
                            .send((
                                format!("{}", FlipHashXXH3128),
                                thread_experiment.accumulate(&FlipHashXXH3128, STEP_SIZE),
                            ))
                            .unwrap();
                    }
                    Algorithm::JumpHash => {
                        thread_tx
                            .send((
                                format!("{}", JumpHash),
                                thread_experiment.accumulate(&JumpHash, STEP_SIZE),
                            ))
                            .unwrap();
                    }
                }
            }
        });
    }

    let mut accumulators = HashMap::new();
    let mut results_by_algo = HashMap::new();
    let mut next_plot_num_iterations = 0;
    for (algo, step_accumulator) in rx {
        let algo_accumulator = accumulators
            .entry(algo.clone())
            .or_insert_with(|| experiment.new_accumulator());
        algo_accumulator.merge(&step_accumulator);

        let result = ExperimentResult {
            algo: algo.clone(),
            result: experiment.result(algo_accumulator),
        };
        serde_json::to_writer(output.by_ref(), &result).unwrap();
        output.write_all("\n".as_bytes()).unwrap();
        output.flush().unwrap();

        results_by_algo
            .entry(algo.clone())
            .or_insert_with(Vec::new)
            .push(result.result);

        if algo_accumulator.num_iterations() >= next_plot_num_iterations {
            next_plot_num_iterations += PLOT_EVERY;
            println!("Plotting...");
            experiment.plot(&results_by_algo).write_image(
                &svg_path,
                ImageFormat::SVG,
                1000,
                600,
                1.0,
            );
        }
        println!(
            "Processed {:e} keys for {}",
            algo_accumulator.num_iterations(),
            algo
        );
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct ExperimentResult<R> {
    algo: String,
    #[serde(flatten)]
    result: R,
}
