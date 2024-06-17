# FlipHash

[Paper](https://arxiv.org/abs/2402.17549) | [Documentation](https://docs.rs/fliphash/0.1.0/fliphash/) | [Crate](https://crates.io/crates/fliphash)

FlipHash is a consistent range-hashing function that hashes an integer
`key` into a value of `..=range_end`, where `range_end` is parameterized.
It is:
- __regular__ (i.e., uniform, balanced): it distributes the keys evenly over the hash values of the range,
- __monotone__ (i.e., stable): when varying the range, key hashes are not shuffled across the values that stay within the range, and keys can only be remapped from a hash value now outside of the range (if the range is narrowed), or to a hash value previously outside of the range (if the range is enlarged),
- __fast__: it has a low computational cost, and constant-time complexity.

## Usage

```rs
use fliphash::fliphash_64;
let hash = fliphash_64(10427592028180905159, ..=17);
assert!((..=17).contains(&hash));
```

## Regularity

The following code snippet illustrates the regularity of FlipHash.

With a large enough number of distinct keys, the numbers of occurrences of
the hash values of `range` are relatively close to one another.

```rs
use fliphash::fliphash_64;
let mut hash_counts = [0_u64; 18];
// Hash a lot of keys; they could be picked randomly.
for key in 0_u64..2_000_000_u64 {
    let hash = fliphash_64(key, ..=17);
    hash_counts[hash as usize] += 1;
}
let (min_count, max_count) = (
    *hash_counts.iter().min().unwrap() as f64,
    *hash_counts.iter().max().unwrap() as f64,
);
let relative_difference = (max_count - min_count) / min_count;
assert!(relative_difference < 0.01);
```

## Monotonicity

The following code snippet illustrates the monotonicity, i.e., the
stability, of FlipHash.

Given a key, when making the range larger, either the hash of the key is unchanged or it gets a new value that the previous range does not contain.

```rs
use fliphash::fliphash_64;
let key = 10427592028180905159;
let mut previous_hash = 0;
for range_end in 1..1000 {
    let hash = fliphash_64(key, ..=range_end);
    assert!(hash == previous_hash || hash == range_end);
    previous_hash = hash;
}
```

## Performance

FlipHash has constant average and worst-case time complexity.

As a comparison, [Jump Consistent Hash](https://arxiv.org/abs/1406.2294) has a time
complexity that is logarithmic in the width of the range.

### Evaluation wall times

On an Intel Xeon Platinum 8375C CPU @ 2.9GHz.

| Range           | FlipHash | JumpHash |
| --------------- | -------- | -------- |
| `..=10`         | 5.9 ns   | 8.2 ns   |
| `..=1000`       | 4.7 ns   | 25 ns    |
| `..=1000000`    | 5.5 ns   | 45 ns    |
| `..=1000000000` | 6.4 ns   | 69 ns    |
