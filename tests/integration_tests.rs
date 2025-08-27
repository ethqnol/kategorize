use kategorize::{KModes, KPrototypes, InitMethod, MixedValue};
use ndarray::Array2;

#[test]
fn test_kmodes_iris_like_data() {
    // Create categorical data similar to iris dataset
    let data = Array2::from_shape_vec(
        (12, 3),
        vec![
            "small", "red", "round",
            "small", "red", "round", 
            "small", "blue", "round",
            "small", "blue", "round",
            "medium", "red", "oval",
            "medium", "red", "oval",
            "medium", "blue", "oval",
            "medium", "blue", "oval",
            "large", "green", "square",
            "large", "green", "square",
            "large", "yellow", "square",
            "large", "yellow", "square",
        ]
    ).unwrap();

    let kmodes = KModes::new(3)
        .random_state(42)
        .n_init(5)
        .max_iter(50)
        .verbose(false);

    let result = kmodes.fit(data.view()).unwrap();
    
    // Check basic properties
    assert_eq!(result.labels.len(), 12);
    assert_eq!(result.centroids.nrows(), 3);
    assert_eq!(result.centroids.ncols(), 3);
    assert!(result.converged);
    assert!(result.inertia >= 0.0);
    
    // Check that we have 3 different clusters
    let unique_labels: std::collections::HashSet<_> = result.labels.iter().collect();
    assert_eq!(unique_labels.len(), 3);
}

#[test]
fn test_kmodes_single_cluster() {
    // All data points are identical
    let data = Array2::from_shape_vec(
        (5, 2),
        vec!["A", "X", "A", "X", "A", "X", "A", "X", "A", "X"]
    ).unwrap();

    let kmodes = KModes::new(1)
        .random_state(42)
        .n_init(3)
        .max_iter(10);

    let result = kmodes.fit(data.view()).unwrap();
    
    // All points should be in cluster 0
    assert!(result.labels.iter().all(|&label| label == 0));
    assert_eq!(result.centroids[[0, 0]], "A");
    assert_eq!(result.centroids[[0, 1]], "X");
    assert_eq!(result.inertia, 0.0); // Perfect clustering
}

#[test]
fn test_kmodes_different_init_methods() {
    let data = Array2::from_shape_vec(
        (8, 2),
        vec!["A", "X", "A", "X", "B", "Y", "B", "Y", 
             "C", "Z", "C", "Z", "D", "W", "D", "W"]
    ).unwrap();

    for init_method in [InitMethod::Random, InitMethod::Huang, InitMethod::Cao] {
        let kmodes = KModes::new(2)
            .init_method(init_method)
            .random_state(42)
            .n_init(3)
            .max_iter(50);

        let result = kmodes.fit(data.view());
        assert!(result.is_ok(), "Failed with init method {:?}", init_method);
        
        let result = result.unwrap();
        assert_eq!(result.labels.len(), 8);
        assert_eq!(result.centroids.nrows(), 2);
    }
}

#[test]
fn test_kprototypes_mixed_data() {
    let data = Array2::from_shape_vec(
        (6, 3),
        vec![
            MixedValue::Categorical("A"), MixedValue::Categorical("X"), MixedValue::Numerical(1.0),
            MixedValue::Categorical("A"), MixedValue::Categorical("X"), MixedValue::Numerical(1.5),
            MixedValue::Categorical("B"), MixedValue::Categorical("Y"), MixedValue::Numerical(5.0),
            MixedValue::Categorical("B"), MixedValue::Categorical("Y"), MixedValue::Numerical(5.5),
            MixedValue::Categorical("C"), MixedValue::Categorical("Z"), MixedValue::Numerical(10.0),
            MixedValue::Categorical("C"), MixedValue::Categorical("Z"), MixedValue::Numerical(10.5),
        ]
    ).unwrap();

    let kproto = KPrototypes::new(3, vec![0, 1], vec![2])
        .random_state(42)
        .gamma(1.0)
        .n_init(3)
        .max_iter(50);

    let result = kproto.fit(data.view(), vec![0, 1], vec![2]).unwrap();
    
    assert_eq!(result.labels.len(), 6);
    assert_eq!(result.centroids.nrows(), 3);
    assert_eq!(result.centroids.ncols(), 3);
    assert_eq!(result.categorical_indices, vec![0, 1]);
    assert_eq!(result.numerical_indices, vec![2]);
    
    // Check that we have reasonable cluster assignments
    let unique_labels: std::collections::HashSet<_> = result.labels.iter().collect();
    assert!(unique_labels.len() <= 3);
}

#[test]
fn test_kprototypes_gamma_effect() {
    let data = Array2::from_shape_vec(
        (4, 2),
        vec![
            MixedValue::Categorical("A"), MixedValue::Numerical(1.0),
            MixedValue::Categorical("A"), MixedValue::Numerical(100.0),
            MixedValue::Categorical("B"), MixedValue::Numerical(1.0),
            MixedValue::Categorical("B"), MixedValue::Numerical(100.0),
        ]
    ).unwrap();

    // Test with different gamma values
    for gamma in [0.1, 1.0, 10.0] {
        let kproto = KPrototypes::new(2, vec![0], vec![1])
            .gamma(gamma)
            .random_state(42)
            .n_init(3)
            .max_iter(50);

        let result = kproto.fit(data.view(), vec![0], vec![1]);
        assert!(result.is_ok(), "Failed with gamma {}", gamma);
    }
}

#[test]
fn test_basic_functionality_check() {
    // Test that both fit and fit_predict work without errors
    let data = Array2::from_shape_vec(
        (4, 2),
        vec!["A", "X", "A", "X", "B", "Y", "B", "Y"]
    ).unwrap();

    let kmodes = KModes::new(2)
        .random_state(42)
        .n_init(1)
        .max_iter(10);

    // Both should work without error
    let full_result = kmodes.fit(data.view()).unwrap();
    let labels_result = kmodes.fit_predict(data.view()).unwrap();

    // Basic sanity checks
    assert_eq!(full_result.labels.len(), 4);
    assert_eq!(labels_result.len(), 4);
    assert!(full_result.labels.iter().all(|&l| l < 2));
    assert!(labels_result.iter().all(|&l| l < 2));
}

#[test]
fn test_error_conditions() {
    let data = Array2::from_shape_vec(
        (2, 2),
        vec!["A", "X", "B", "Y"]
    ).unwrap();

    // Test too many clusters
    let kmodes = KModes::new(5);
    assert!(kmodes.fit(data.view()).is_err());

    // Test empty data
    let empty_data = Array2::from_shape_vec((0, 0), Vec::<&str>::new()).unwrap();
    let kmodes = KModes::new(1);
    assert!(kmodes.fit(empty_data.view()).is_err());

    // Test zero clusters
    let kmodes = KModes::new(0);
    assert!(kmodes.fit(data.view()).is_err());
}

#[test]
fn test_cluster_quality_metrics() {
    let data = Array2::from_shape_vec(
        (9, 2),
        vec![
            "A", "X", "A", "X", "A", "X", // Group 1
            "B", "Y", "B", "Y", "B", "Y", // Group 2  
            "C", "Z", "C", "Z", "C", "Z", // Group 3
        ]
    ).unwrap();

    let kmodes = KModes::new(3)
        .random_state(42)
        .n_init(10)
        .max_iter(100);

    let result = kmodes.fit(data.view()).unwrap();
    
    // With perfect separation, inertia should be 0
    assert_eq!(result.inertia, 0.0);
    
    // Each cluster should have exactly 3 points
    let mut cluster_sizes = vec![0; 3];
    for &label in result.labels.iter() {
        cluster_sizes[label] += 1;
    }
    assert_eq!(cluster_sizes, vec![3, 3, 3]);
}