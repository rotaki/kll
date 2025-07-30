# KLL Sketch - Generic Rust Implementation

This is a generic Rust implementation of the KLL (K-th Largest/K-th Quantile) streaming quantiles sketch algorithm.

## Overview

The KLL sketch is a data structure for approximating quantiles in a data stream with limited memory. It provides:

- O(k log n) space complexity
- Fast updates
- Accurate quantile estimation
- Support for merging sketches
- **Generic type support** - works with any type that implements `PartialOrd` and `Clone`

## Usage

```rust
use kll2::Sketch;

fn main() {
    // Works with floating point numbers
    let mut float_sketch = Sketch::<f64>::new(200);
    for i in 0..10000 {
        float_sketch.update(i as f64);
    }
    
    // Works with integers
    let mut int_sketch = Sketch::<i32>::new(200);
    for i in 0..10000 {
        int_sketch.update(i);
    }
    
    // Works with strings
    let mut string_sketch = Sketch::<String>::new(100);
    string_sketch.update("apple".to_string());
    string_sketch.update("banana".to_string());
    string_sketch.update("cherry".to_string());
    
    // Works with custom types (that implement PartialOrd and Clone)
    let mut vec_sketch = Sketch::<Vec<u8>>::new(100);
    vec_sketch.update(vec![1, 2, 3]);
    vec_sketch.update(vec![4, 5, 6]);
    
    // Get statistics
    println!("Count: {}", float_sketch.count());
    println!("Rank of 5000: {}", float_sketch.rank(5000.0));
    
    // Get CDF and query percentiles
    let cdf = float_sketch.cdf();
    println!("50th percentile: {}", cdf.query(0.5));
    println!("90th percentile: {}", cdf.query(0.9));
}
```

## Features

- Core KLL sketch implementation with generic type support
- CDF computation
- Sketch merging
- Support for any type that implements `PartialOrd` and `Clone`

## Type Requirements

The generic type `T` must implement:
- `PartialOrd` - for ordering/comparison operations
- `Clone` - for storing and manipulating values in the data structure

## API

- `Sketch::new(k)` - Create a new sketch with parameter k (controls accuracy/space tradeoff)
- `sketch.update(value)` - Add a value to the sketch
- `sketch.rank(value)` - Get the approximate rank of a value
- `sketch.count()` - Get the total number of items added
- `sketch.quantile(value)` - Get the quantile (0-1) of a value
- `sketch.cdf()` - Get the cumulative distribution function
- `sketch.merge(&other)` - Merge another sketch into this one

## CDF Operations

- `cdf.query(p)` - Get the value at quantile p (0-1)
- `cdf.quantile(value)` - Get the quantile (0-1) of a value

## Reference

- Based on the paper: [Optimal Quantile Approximation in Streams](http://arxiv.org/pdf/1603.05346v1.pdf)
- Go implementation: [Go KLL Sketch](https://github.com/dgryski/go-kll)
