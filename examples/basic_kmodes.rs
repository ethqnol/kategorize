//! Basic k-modes clustering example
//! 
//! This example demonstrates how to perform k-modes clustering on categorical data
//! using different initialization methods and parameters.

use kategorize::{KModes, InitMethod};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create some sample categorical data
    // This represents a small dataset with 3 features (size, color, shape)
    let data = Array2::from_shape_vec(
        (12, 3), 
        vec![
            // Small red items
            "small", "red", "circle",
            "small", "red", "circle", 
            "small", "red", "square",
            
            // Medium blue items
            "medium", "blue", "circle",
            "medium", "blue", "triangle",
            "medium", "blue", "triangle",
            
            // Large green items
            "large", "green", "square",
            "large", "green", "square",
            "large", "green", "triangle",
            
            // Some mixed items
            "small", "blue", "circle",
            "medium", "red", "square", 
            "large", "red", "triangle",
        ]
    )?;

    println!("Sample data shape: {:?}", data.dim());
    println!("First few rows:");
    for i in 0..3 {
        println!("  {:?}", data.row(i).to_vec());
    }
    println!();

    // Perform k-modes clustering with different settings
    
    // Example 1: Basic clustering with Huang initialization
    println!("=== Example 1: Basic K-modes with Huang initialization ===");
    let kmodes = KModes::new(3)
        .init_method(InitMethod::Huang)
        .random_state(42)
        .n_init(5)
        .max_iter(100)
        .verbose(false);

    let result = kmodes.fit(data.view())?;
    
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.n_iter);
    println!("Final inertia: {:.4}", result.inertia);
    println!("Cluster assignments: {:?}", result.labels.to_vec());
    println!("Centroids:");
    for (i, centroid) in result.centroids.rows().into_iter().enumerate() {
        println!("  Cluster {}: {:?}", i, centroid.to_vec());
    }
    println!();

    // Example 2: Compare different initialization methods
    println!("=== Example 2: Comparing initialization methods ===");
    let init_methods = [
        ("Random", InitMethod::Random),
        ("Huang", InitMethod::Huang), 
        ("Cao", InitMethod::Cao),
    ];

    for (name, method) in &init_methods {
        let kmodes = KModes::new(3)
            .init_method(*method)
            .random_state(42)
            .n_init(1) // Single run for comparison
            .max_iter(50);

        let result = kmodes.fit(data.view())?;
        println!("{:6} init - Inertia: {:.4}, Iterations: {}", 
                 name, result.inertia, result.n_iter);
    }
    println!();

    // Example 3: Effect of number of clusters
    println!("=== Example 3: Effect of number of clusters ===");
    for k in 2..=5 {
        let kmodes = KModes::new(k)
            .init_method(InitMethod::Huang)
            .random_state(42)
            .n_init(3)
            .max_iter(50);

        let result = kmodes.fit(data.view())?;
        println!("k={}: Inertia={:.4}, Converged={}", k, result.inertia, result.converged);
    }
    println!();

    // Example 4: Using fit_predict for just getting labels
    println!("=== Example 4: Using fit_predict ===");
    let kmodes = KModes::new(3)
        .init_method(InitMethod::Huang)
        .random_state(42);
        
    let labels = kmodes.fit_predict(data.view())?;
    println!("Just the cluster labels: {:?}", labels.to_vec());
    
    // Group data points by cluster
    println!("Data points by cluster:");
    for cluster_id in 0..3 {
        let points_in_cluster: Vec<_> = labels.iter()
            .enumerate()
            .filter_map(|(i, &label)| {
                if label == cluster_id {
                    Some(data.row(i).to_vec())
                } else {
                    None
                }
            })
            .collect();
        
        println!("  Cluster {}: {} points", cluster_id, points_in_cluster.len());
        for point in points_in_cluster {
            println!("    {:?}", point);
        }
    }

    Ok(())
}