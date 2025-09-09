//! Example demonstrating Jaccard distance for k-modes clustering
//!
//! This example shows how to use the Jaccard distance metric with k-modes clustering.
//! Jaccard distance is particularly useful when dealing with categorical data where
//! the relationship between categories can be viewed as set membership.

use kategorize::{KModes, DistanceMetric, InitMethod};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Jaccard Distance K-modes Clustering Example ===\n");

    // Create sample categorical data
    // This represents different users' preferences or characteristics
    let data = Array2::from_shape_vec(
        (8, 3),
        vec![
            // User preferences: [Music_Genre, Food_Type, Sport]
            "Rock", "Italian", "Soccer",      // User 1
            "Rock", "Italian", "Basketball",  // User 2
            "Pop", "Chinese", "Tennis",       // User 3
            "Pop", "Chinese", "Swimming",     // User 4
            "Jazz", "Mexican", "Soccer",      // User 5
            "Jazz", "Mexican", "Basketball",  // User 6
            "Rock", "Italian", "Tennis",      // User 7
            "Pop", "Chinese", "Soccer",       // User 8
        ],
    )?;

    println!("Sample data (User preferences):");
    println!("Features: [Music_Genre, Food_Type, Sport]");
    for (i, row) in data.rows().into_iter().enumerate() {
        println!("User {}: [{}, {}, {}]", i + 1, row[0], row[1], row[2]);
    }
    println!();

    // Example 1: Compare different distance metrics
    println!("=== Comparing Distance Metrics ===");
    
    let distance_metrics = [
        ("Matching", DistanceMetric::Matching),
        ("Hamming", DistanceMetric::Hamming),
        ("Jaccard", DistanceMetric::Jaccard),
    ];

    for (name, metric) in &distance_metrics {
        println!("--- {} Distance ---", name);
        
        let kmodes = KModes::new(3)
            .distance_metric(*metric)
            .init_method(InitMethod::Huang)
            .random_state(42)
            .n_init(5)
            .max_iter(50);

        let result = kmodes.fit(data.view())?;
        
        println!("Converged: {}", result.converged);
        println!("Iterations: {}", result.n_iter);
        println!("Inertia: {:.2}", result.inertia);
        
        // Show cluster assignments
        println!("Cluster assignments:");
        for (user_id, &cluster_id) in result.labels.iter().enumerate() {
            println!("  User {} -> Cluster {}", user_id + 1, cluster_id);
        }
        
        // Show cluster centroids
        println!("Cluster centroids:");
        for (cluster_id, centroid) in result.centroids.rows().into_iter().enumerate() {
            println!("  Cluster {}: [{}, {}, {}]", 
                     cluster_id, centroid[0], centroid[1], centroid[2]);
        }
        println!();
    }

    // Example 2: Demonstrate why Jaccard distance is useful
    println!("=== Why Jaccard Distance? ===");
    
    // Create data where set-based similarity matters
    let set_like_data = Array2::from_shape_vec(
        (6, 4),
        vec![
            // Each row represents tags or categories associated with an item
            "tag1", "tag2", "tag3", "tag1",  // Item 1: {tag1, tag2, tag3} 
            "tag1", "tag2", "tag1", "tag2",  // Item 2: {tag1, tag2}
            "tag4", "tag5", "tag6", "tag4",  // Item 3: {tag4, tag5, tag6}
            "tag4", "tag5", "tag4", "tag5",  // Item 4: {tag4, tag5}
            "tag1", "tag3", "tag7", "tag1",  // Item 5: {tag1, tag3, tag7}
            "tag2", "tag3", "tag8", "tag2",  // Item 6: {tag2, tag3, tag8}
        ],
    )?;

    println!("Tag-based data (showing how Jaccard handles set-like relationships):");
    for (i, row) in set_like_data.rows().into_iter().enumerate() {
        let unique_tags: std::collections::HashSet<_> = row.iter().collect();
        println!("Item {}: {:?}", i + 1, unique_tags);
    }
    println!();

    let jaccard_kmodes = KModes::new(2)
        .distance_metric(DistanceMetric::Jaccard)
        .random_state(42)
        .n_init(3)
        .max_iter(50);

    let jaccard_result = jaccard_kmodes.fit(set_like_data.view())?;
    
    println!("Jaccard clustering results:");
    println!("Cluster assignments:");
    for (item_id, &cluster_id) in jaccard_result.labels.iter().enumerate() {
        println!("  Item {} -> Cluster {}", item_id + 1, cluster_id);
    }
    
    println!("\nCluster centroids:");
    for (cluster_id, centroid) in jaccard_result.centroids.rows().into_iter().enumerate() {
        let unique_elements: std::collections::HashSet<_> = centroid.iter().collect();
        println!("  Cluster {}: {:?}", cluster_id, unique_elements);
    }

    println!("\n=== Summary ===");
    println!("• Matching Distance: Simple binary comparison (0 or 1 per feature)");
    println!("• Hamming Distance: Normalized matching distance (proportion of mismatches)"); 
    println!("• Jaccard Distance: Set-based similarity, good for:");
    println!("  - Data with repeated values that should be treated as sets");
    println!("  - Categorical data where intersection/union relationships matter");
    println!("  - Tag-based or multi-label categorical data");

    Ok(())
}