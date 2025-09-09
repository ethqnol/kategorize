# Kategorize

[![Crates.io](https://img.shields.io/crates/v/kategorize.svg)](https://crates.io/crates/kategorize)
[![Documentation](https://docs.rs/kategorize/badge.svg)](https://docs.rs/kategorize)
[![License](https://img.shields.io/crates/l/kategorize.svg)](https://github.com/ethqnol/kategorize#license)

A fast, memory-efficient Rust implementation of k-modes and k-prototypes clustering algorithms for categorical and mixed data.

## Features

- **K-modes clustering**: Designed specifically for categorical data
- **K-prototypes clustering**: Handles mixed categorical and numerical data  
- **Multiple initialization methods**: Huang, Cao, and random initialization
- **Multiple distance metrics**: Matching, Hamming, and Jaccard distance support
- **Parallel processing**: Multi-threaded execution for better performance
- **Memory efficient**: Optimized data structures and algorithms
- **Comprehensive**: Full-featured with proper error handling and validation
- **Well tested**: Extensive unit tests, integration tests, and benchmarks

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
kategorize = "0.1"
```

### Basic K-modes Clustering

```rust
use kategorize::{KModes, InitMethod, DistanceMetric};
use ndarray::Array2;

// Create categorical data
let data = Array2::from_shape_vec((6, 2), vec![
    "A", "X", "A", "X", "B", "Y", 
    "B", "Y", "C", "Z", "C", "Z"
]).unwrap();

// Configure and run k-modes clustering
let kmodes = KModes::new(3)
    .init_method(InitMethod::Huang)
    .distance_metric(DistanceMetric::Jaccard)  // Choose distance metric
    .max_iter(100)
    .n_init(10)
    .random_state(42);

let result = kmodes.fit(data.view()).unwrap();

println!("Cluster assignments: {:?}", result.labels);
println!("Centroids: {:?}", result.centroids);
println!("Converged: {}", result.converged);
```

### K-prototypes for Mixed Data

```rust
use kategorize::{KPrototypes, MixedValue};
use ndarray::Array2;

// Create mixed categorical and numerical data
let data = Array2::from_shape_vec((4, 3), vec![
    MixedValue::Categorical("A"), MixedValue::Categorical("X"), MixedValue::Numerical(1.0),
    MixedValue::Categorical("A"), MixedValue::Categorical("X"), MixedValue::Numerical(2.0),
    MixedValue::Categorical("B"), MixedValue::Categorical("Y"), MixedValue::Numerical(10.0),
    MixedValue::Categorical("B"), MixedValue::Categorical("Y"), MixedValue::Numerical(11.0),
]).unwrap();

let kprototypes = KPrototypes::new(2, vec![0, 1], vec![2])  // categorical: [0,1], numerical: [2]
    .gamma(1.0)  // weight for numerical vs categorical features
    .random_state(42);

let result = kprototypes.fit(data.view(), vec![0, 1], vec![2]).unwrap();
println!("Mixed data clustering result: {:?}", result.labels);
```

## Algorithms

### K-modes
K-modes extends k-means clustering to categorical data by:
- Using simple matching dissimilarity instead of Euclidean distance
- Replacing means with modes (most frequent values) for centroid updates
- Supporting various initialization strategies for better convergence

### K-prototypes  
K-prototypes combines k-modes and k-means to handle mixed data by:
- Computing separate distances for categorical and numerical features
- Using a γ (gamma) parameter to balance the two distance components
- Updating centroids using modes for categorical and means for numerical features

## Initialization Methods

- **Huang**: Density-based initialization selecting most frequent data points
- **Cao**: Improved initialization maximizing dissimilarity between initial centroids  
- **Random**: Randomly selected data points as initial centroids

## Distance Metrics

- **Matching**: Simple matching dissimilarity (0 for match, 1 for mismatch)
- **Hamming**: Normalized matching distance (proportion of mismatches)
- **Jaccard**: Set-based similarity, ideal for data with repeated values or tag-like categories

## Performance

Kategorize is designed for performance with:
- Parallel processing using Rayon for multiple initialization runs
- Efficient data structures minimizing memory allocation
- Optimized distance calculations for categorical data
- Early convergence detection

## Examples

Check out the [examples](examples/) directory for comprehensive usage patterns:

- [`basic_kmodes.rs`](examples/basic_kmodes.rs) - Basic k-modes clustering
- [`kprototypes_mixed_data.rs`](examples/kprototypes_mixed_data.rs) - Mixed data clustering  
- [`advanced_usage.rs`](examples/advanced_usage.rs) - Parameter tuning and optimization
- [`jaccard_distance.rs`](examples/jaccard_distance.rs) - Jaccard distance metric usage

Run examples with:
```bash
cargo run --example basic_kmodes
cargo run --example kprototypes_mixed_data
cargo run --example advanced_usage
cargo run --example jaccard_distance
```

## Benchmarks

Run benchmarks to see performance characteristics:

```bash
cargo bench
```

This will run comprehensive benchmarks testing:
- Different data sizes and cluster counts
- Initialization method performance
- Scaling characteristics
- Parameter sensitivity

## API Documentation

For detailed API documentation, visit [docs.rs/kategorize](https://docs.rs/kategorize) or run:

```bash
cargo doc --open
```

## Comparison with Python kmodes

Kategorize provides similar functionality to the popular Python [kmodes](https://github.com/nicodv/kmodes) library but with:

- **Better performance**: Rust's zero-cost abstractions and memory safety
- **Type safety**: Compile-time guarantees about data types and dimensions
- **Parallel processing**: Built-in support for multi-threading
- **Memory efficiency**: Lower memory footprint and no GIL limitations

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where contributions would be particularly valuable:
- Additional initialization methods
- More distance metrics for categorical data
- Incremental/online clustering algorithms
- Integration with other ML libraries

## License

This project is licensed under the [MIT License](LICENSE-MIT).

## Citation

If you use Kategorize in your research, please cite:

```bibtex
@software{kategorize,
  title = {Kategorize: K-modes and K-prototypes clustering for Rust},
  author = {Wu, Ethan},
  year = {2024},
  url = {https://github.com/ethqnol/kategorize}
}
```

## References

1. Huang, Z. (1997). A Fast Clustering Algorithm to Cluster Very Large Categorical Data Sets in Data Mining. SIGMOD Record.
2. Huang, Z. (1998). Extensions to the k-Means Algorithm for Clustering Large Data Sets with Categorical Values. Data Mining and Knowledge Discovery.  
3. Cao, F., Liang, J, Bai, L. (2009). A new initialization method for categorical data clustering. Expert Systems with Applications.