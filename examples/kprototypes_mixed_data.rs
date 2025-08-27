//! K-prototypes clustering for mixed categorical and numerical data
//! 
//! This example shows how to use k-prototypes to cluster data that contains
//! both categorical and numerical features.

use kategorize::{KPrototypes, MixedValue};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create mixed data representing customers with both categorical and numerical attributes
    // Features: [segment, region, age, income, satisfaction_score]
    //           [cat,     cat,    num, num,   num]
    let data = Array2::from_shape_vec(
        (15, 5),
        vec![
            // Young urban professionals
            MixedValue::Categorical("professional".to_string()), MixedValue::Categorical("urban".to_string()), MixedValue::Numerical(28.0), MixedValue::Numerical(75000.0), MixedValue::Numerical(8.5),
            MixedValue::Categorical("professional".to_string()), MixedValue::Categorical("urban".to_string()), MixedValue::Numerical(32.0), MixedValue::Numerical(82000.0), MixedValue::Numerical(7.8),
            MixedValue::Categorical("professional".to_string()), MixedValue::Categorical("urban".to_string()), MixedValue::Numerical(29.0), MixedValue::Numerical(78000.0), MixedValue::Numerical(8.2),
            
            // Middle-aged suburban families  
            MixedValue::Categorical("family".to_string()), MixedValue::Categorical("suburban".to_string()), MixedValue::Numerical(45.0), MixedValue::Numerical(95000.0), MixedValue::Numerical(7.2),
            MixedValue::Categorical("family".to_string()), MixedValue::Categorical("suburban".to_string()), MixedValue::Numerical(42.0), MixedValue::Numerical(88000.0), MixedValue::Numerical(7.5),
            MixedValue::Categorical("family".to_string()), MixedValue::Categorical("suburban".to_string()), MixedValue::Numerical(47.0), MixedValue::Numerical(102000.0), MixedValue::Numerical(6.9),
            MixedValue::Categorical("family".to_string()), MixedValue::Categorical("suburban".to_string()), MixedValue::Numerical(44.0), MixedValue::Numerical(92000.0), MixedValue::Numerical(7.1),
            
            // Rural retirees
            MixedValue::Categorical("retired".to_string()), MixedValue::Categorical("rural".to_string()), MixedValue::Numerical(68.0), MixedValue::Numerical(45000.0), MixedValue::Numerical(6.2),
            MixedValue::Categorical("retired".to_string()), MixedValue::Categorical("rural".to_string()), MixedValue::Numerical(72.0), MixedValue::Numerical(38000.0), MixedValue::Numerical(5.8),
            MixedValue::Categorical("retired".to_string()), MixedValue::Categorical("rural".to_string()), MixedValue::Numerical(65.0), MixedValue::Numerical(42000.0), MixedValue::Numerical(6.0),
            
            // Young students
            MixedValue::Categorical("student".to_string()), MixedValue::Categorical("urban".to_string()), MixedValue::Numerical(22.0), MixedValue::Numerical(25000.0), MixedValue::Numerical(7.8),
            MixedValue::Categorical("student".to_string()), MixedValue::Categorical("urban".to_string()), MixedValue::Numerical(20.0), MixedValue::Numerical(20000.0), MixedValue::Numerical(8.1),
            MixedValue::Categorical("student".to_string()), MixedValue::Categorical("suburban".to_string()), MixedValue::Numerical(23.0), MixedValue::Numerical(22000.0), MixedValue::Numerical(7.5),
            
            // Mixed cases
            MixedValue::Categorical("professional".to_string()), MixedValue::Categorical("rural".to_string()), MixedValue::Numerical(35.0), MixedValue::Numerical(65000.0), MixedValue::Numerical(7.0),
            MixedValue::Categorical("family".to_string()), MixedValue::Categorical("urban".to_string()), MixedValue::Numerical(38.0), MixedValue::Numerical(85000.0), MixedValue::Numerical(7.3),
        ]
    )?;

    println!("Mixed data shape: {:?}", data.dim());
    println!("Features: [segment, region, age, income, satisfaction_score]");
    println!("Data types: [cat, cat, num, num, num]");
    println!();

    // Define which features are categorical vs numerical
    let categorical_indices = vec![0, 1];  // segment, region
    let numerical_indices = vec![2, 3, 4]; // age, income, satisfaction_score

    // Example 1: Basic k-prototypes clustering
    println!("=== Example 1: Basic K-prototypes clustering ===");
    
    let kproto = KPrototypes::new(4, categorical_indices.clone(), numerical_indices.clone())
        .gamma(1.0)  // Weight balance between categorical and numerical features
        .random_state(42)
        .n_init(3)
        .max_iter(100);

    let result = kproto.fit(data.view(), categorical_indices.clone(), numerical_indices.clone())?;
    
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.n_iter); 
    println!("Final inertia: {:.4}", result.inertia);
    println!("Cluster assignments: {:?}", result.labels.to_vec());
    println!();

    println!("Cluster centroids (prototypes):");
    for (i, centroid) in result.centroids.rows().into_iter().enumerate() {
        println!("  Cluster {}:", i);
        let values: Vec<String> = centroid.iter().map(|v| {
            match v {
                MixedValue::Categorical(s) => format!("'{}'", s),
                MixedValue::Numerical(n) => format!("{:.1}", n),
            }
        }).collect();
        println!("    [{}]", values.join(", "));
    }
    println!();

    // Example 2: Effect of gamma parameter
    println!("=== Example 2: Effect of gamma parameter ===");
    let gammas = [0.1, 0.5, 1.0, 2.0, 5.0];
    
    for &gamma in &gammas {
        let kproto = KPrototypes::new(4, categorical_indices.clone(), numerical_indices.clone())
            .gamma(gamma)
            .random_state(42)
            .n_init(1)
            .max_iter(50);

        let result = kproto.fit(data.view(), categorical_indices.clone(), numerical_indices.clone())?;
        println!("Gamma {:.1}: Inertia={:.4}, Converged={}", gamma, result.inertia, result.converged);
    }
    println!();

    // Example 3: Analyze clusters by their composition
    println!("=== Example 3: Cluster composition analysis ===");
    
    let kproto = KPrototypes::new(3, categorical_indices.clone(), numerical_indices.clone())
        .gamma(1.0)
        .random_state(42)
        .n_init(5);

    let result = kproto.fit(data.view(), categorical_indices.clone(), numerical_indices)?;
    
    for cluster_id in 0..3 {
        let points_in_cluster: Vec<_> = result.labels.iter()
            .enumerate()
            .filter_map(|(i, &label)| {
                if label == cluster_id {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        
        println!("Cluster {} ({} points):", cluster_id, points_in_cluster.len());
        
        // Show the centroid/prototype
        let centroid = result.centroids.row(cluster_id);
        print!("  Prototype: ");
        for (j, value) in centroid.iter().enumerate() {
            let feature_name = match j {
                0 => "segment",
                1 => "region", 
                2 => "age",
                3 => "income",
                4 => "satisfaction",
                _ => "feature",
            };
            
            match value {
                MixedValue::Categorical(s) => print!("{}='{}' ", feature_name, s),
                MixedValue::Numerical(n) => print!("{}={:.1} ", feature_name, n),
            }
        }
        println!();
        
        // Show individual data points
        println!("  Data points:");
        for &point_idx in &points_in_cluster {
            let row = data.row(point_idx);
            print!("    ");
            for value in row {
                match value {
                    MixedValue::Categorical(s) => print!("{:12} ", s),
                    MixedValue::Numerical(n) => print!("{:8.1}     ", n),
                }
            }
            println!();
        }
        println!();
    }

    Ok(())
}