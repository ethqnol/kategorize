//! Advanced usage patterns for kategorize clustering
//! 
//! This example demonstrates advanced features like custom distance metrics,
//! parameter tuning, and performance optimization techniques.

use kategorize::{KModes, InitMethod};
use ndarray::Array2;
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate a larger synthetic dataset for demonstration
    let data = generate_synthetic_data(1000, 8, 15);
    println!("Generated synthetic data: {} samples, {} features", data.nrows(), data.ncols());
    println!();

    // Example 1: Parameter tuning for optimal number of clusters
    println!("=== Example 1: Finding optimal number of clusters ===");
    let start = Instant::now();
    
    let k_range = 2..=10;
    let mut inertias = Vec::new();
    
    for k in k_range.clone() {
        let kmodes = KModes::new(k)
            .init_method(InitMethod::Huang)
            .random_state(42)
            .n_init(3)
            .max_iter(100);

        let result = kmodes.fit(data.view())?;
        inertias.push((k, result.inertia));
        println!("k={:2}: inertia={:8.2}, converged={}", k, result.inertia, result.converged);
    }
    
    // Simple elbow method - look for largest decrease in inertia
    let mut best_k = 2;
    let mut max_decrease = 0.0;
    
    for i in 1..inertias.len() {
        let decrease = inertias[i-1].1 - inertias[i].1;
        let decrease_rate = decrease / inertias[i-1].1;
        if decrease_rate > max_decrease {
            max_decrease = decrease_rate;
            best_k = inertias[i].0;
        }
    }
    
    println!("Suggested optimal k: {} (largest relative inertia decrease: {:.1}%)", 
             best_k, max_decrease * 100.0);
    println!("Time taken: {:.2}s", start.elapsed().as_secs_f64());
    println!();

    // Example 2: Comparing initialization strategies with multiple runs
    println!("=== Example 2: Initialization strategy comparison ===");
    let test_data = data.slice(ndarray::s![0..200, ..]).to_owned(); // Smaller subset for faster testing
    
    let strategies = [
        ("Random", InitMethod::Random),
        ("Huang", InitMethod::Huang),
        ("Cao", InitMethod::Cao),
    ];
    
    for (name, method) in &strategies {
        let mut results = Vec::new();
        let start = Instant::now();
        
        // Run multiple times to get statistics
        for seed in 0..5 {
            let kmodes = KModes::new(best_k)
                .init_method(*method)
                .random_state(seed)
                .n_init(1)
                .max_iter(100);
            
            let result = kmodes.fit(test_data.view())?;
            results.push(result.inertia);
        }
        
        let mean_inertia = results.iter().sum::<f64>() / results.len() as f64;
        let min_inertia = results.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_inertia = results.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let std_dev = {
            let variance = results.iter()
                .map(|&x| (x - mean_inertia).powi(2))
                .sum::<f64>() / results.len() as f64;
            variance.sqrt()
        };
        
        println!("{:8} - Mean: {:8.2}, Min: {:8.2}, Max: {:8.2}, Std: {:6.2}, Time: {:5.3}s",
                 name, mean_inertia, min_inertia, max_inertia, std_dev, start.elapsed().as_secs_f64());
    }
    println!();

    // Example 3: Performance optimization with parallel processing
    println!("=== Example 3: Performance optimization ===");
    let large_data = data.slice(ndarray::s![0..500, ..]).to_owned();
    
    // Test with different n_init values (parallel vs sequential)
    let n_init_values = [1, 3, 5, 10];
    
    for &n_init in &n_init_values {
        let start = Instant::now();
        
        let kmodes = KModes::new(best_k)
            .init_method(InitMethod::Huang)
            .random_state(42)
            .n_init(n_init)
            .max_iter(50)
; // Use default parallelization
        
        let result = kmodes.fit(large_data.view())?;
        let elapsed = start.elapsed().as_secs_f64();
        
        println!("n_init={:2}: inertia={:8.2}, time={:5.3}s, time/init={:.3}s",
                 n_init, result.inertia, elapsed, elapsed / n_init as f64);
    }
    println!();

    // Example 4: Cluster quality analysis
    println!("=== Example 4: Cluster quality analysis ===");
    let kmodes = KModes::new(best_k)
        .init_method(InitMethod::Huang)
        .random_state(42)
        .n_init(5)
        .max_iter(100);
    
    let result = kmodes.fit(data.view())?;
    
    // Analyze cluster sizes and balance
    let mut cluster_counts = HashMap::new();
    for &label in result.labels.iter() {
        *cluster_counts.entry(label).or_insert(0) += 1;
    }
    
    println!("Cluster size distribution:");
    let mut sizes: Vec<_> = cluster_counts.iter().collect();
    sizes.sort_by_key(|(k, _)| *k);
    
    let total_points = result.labels.len();
    for (cluster_id, count) in sizes {
        let percentage = *count as f64 / total_points as f64 * 100.0;
        println!("  Cluster {}: {:4} points ({:5.1}%)", cluster_id, count, percentage);
    }
    
    // Check for empty clusters or very imbalanced clustering
    let min_size = cluster_counts.values().min().unwrap_or(&0);
    let max_size = cluster_counts.values().max().unwrap_or(&0);
    let size_ratio = *max_size as f64 / (*min_size).max(1) as f64;
    
    println!("Size balance ratio (max/min): {:.1}", size_ratio);
    if size_ratio > 5.0 {
        println!("Warning: Clusters are highly imbalanced (ratio > 5.0)");
    } else {
        println!("Clusters are reasonably balanced");
    }
    println!();

    // Example 5: Feature importance analysis
    println!("=== Example 5: Feature importance analysis ===");
    
    // Analyze which features contribute most to cluster separation
    let centroids = &result.centroids;
    
    println!("Centroid diversity by feature:");
    for feature_idx in 0..data.ncols() {
        let mut unique_values = std::collections::HashSet::new();
        
        for cluster_idx in 0..centroids.nrows() {
            unique_values.insert(centroids[[cluster_idx, feature_idx]].clone());
        }
        
        let diversity = unique_values.len() as f64 / centroids.nrows() as f64;
        println!("  Feature {}: {:.2} diversity ({} unique values out of {} clusters)",
                 feature_idx, diversity, unique_values.len(), centroids.nrows());
    }

    Ok(())
}

fn generate_synthetic_data(n_samples: usize, n_features: usize, n_categories: usize) -> Array2<String> {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(12345);
    
    let mut data = Vec::with_capacity(n_samples * n_features);
    
    // Generate clustered data with some noise
    for sample_idx in 0..n_samples {
        // Create natural clusters by biasing category selection
        let cluster_bias = sample_idx % 4; // 4 natural clusters
        
        for feature_idx in 0..n_features {
            let category = if rng.gen::<f64>() < 0.7 {
                // Biased selection for clustering
                (cluster_bias * n_categories / 4 + feature_idx + rng.gen_range(0..2)) % n_categories
            } else {
                // Random selection for noise
                rng.gen_range(0..n_categories)
            };
            
            data.push(format!("cat_{}_{}", feature_idx, category));
        }
    }
    
    Array2::from_shape_vec((n_samples, n_features), data).unwrap()
}