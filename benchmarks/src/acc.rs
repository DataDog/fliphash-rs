// Unless explicitly stated otherwise all files in this repository are licensed under the Apache License Version 2.0.
// This product includes software developed at Datadog (https://www.datadoghq.com/).
// Copyright 2024 Datadog, Inc.

use std::{collections::HashMap, fmt::Debug, hash::Hash, marker::PhantomData};

pub(crate) trait Accumulator {
    type Value;
    fn record(&mut self, value: Self::Value);
    fn merge(&mut self, other: &Self);
    fn num_iterations(&self) -> u64;
}

pub(crate) struct NumOccurrences<V> {
    counts: Vec<u64>,
    value_type: PhantomData<V>,
}
impl<V> NumOccurrences<V> {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            counts: vec![0; len],
            value_type: PhantomData,
        }
    }

    pub(crate) fn counts(&self) -> &Vec<u64> {
        &self.counts
    }
}
impl<V> Accumulator for NumOccurrences<V>
where
    usize: TryFrom<V>,
    <usize as TryFrom<V>>::Error: Debug,
{
    type Value = V;

    #[inline]
    fn record(&mut self, value: Self::Value) {
        self.counts[usize::try_from(value).unwrap()] += 1;
    }

    fn merge(&mut self, other: &Self) {
        self.counts
            .iter_mut()
            .zip(other.counts.iter())
            .for_each(|(s, o)| *s += o);
    }

    fn num_iterations(&self) -> u64 {
        self.counts.iter().sum::<u64>()
    }
}

pub(crate) struct NumCooccurrences<V> {
    counts: HashMap<Vec<V>, u64>,
}
impl<V> NumCooccurrences<V>
where
    V: Eq + Hash,
{
    pub(crate) fn new(support: impl IntoIterator<Item = Vec<V>>) -> Self {
        Self {
            counts: support.into_iter().map(|v| (v, 0)).collect(),
        }
    }

    pub(crate) fn counts(&self) -> &HashMap<Vec<V>, u64> {
        &self.counts
    }
}

impl<V> Accumulator for NumCooccurrences<V>
where
    V: Eq + Hash,
{
    type Value = Vec<V>;

    fn record(&mut self, value: Self::Value) {
        *self.counts.get_mut(&value).unwrap() += 1;
    }

    fn merge(&mut self, other: &Self) {
        other
            .counts
            .iter()
            .for_each(|(v, c)| *self.counts.get_mut(v).unwrap() += c)
    }

    fn num_iterations(&self) -> u64 {
        self.counts.values().sum::<u64>()
    }
}
