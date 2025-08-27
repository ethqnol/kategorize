//! K-modes clustering algorithm implementation

use crate::distance::{compute_modes, CategoricalDistance, MatchingDistance};
use crate::error::{Error, Result};
use crate::initialization::{initialize_centroids, InitMethod};
use crate::utils::{
    assign_points_to_centroids, assignments_equal, calculate_cost, get_cluster_indices,
    validate_data, validate_parameters,
};
use ndarray::{Array1, Array2, ArrayView2};
use rand::prelude::*;
use rayon::prelude::*;
use std::hash::Hash;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// K-modes clustering algorithm for categorical data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KModes {
    /// Number of clusters
    pub n_clusters: usize,
    /// Initialization method
    pub init_method: InitMethod,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Number of initialization runs
    pub n_init: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Number of parallel jobs (-1 for all cores)
    pub n_jobs: Option<usize>,
    /// Enable verbose output
    pub verbose: bool,
}

/// Result of k-modes clustering
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KModesResult<T> {
    /// Cluster labels for each data point
    pub labels: Array1<usize>,
    /// Final cluster centroids (modes)
    pub centroids: Array2<T>,
    /// Number of iterations until convergence
    pub n_iter: usize,
    /// Final inertia (total within-cluster distance)
    pub inertia: f64,
    /// Whether the algorithm converged
    pub converged: bool,
}

impl Default for KModes {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            init_method: InitMethod::Huang,
            max_iter: 100,
            tol: 1e-4,
            n_init: 10,
            random_state: None,
            n_jobs: None,
            verbose: false,
        }
    }
}

impl KModes {
    /// Create a new k-modes clusterer with specified number of clusters
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Default::default()
        }
    }

    /// Set the initialization method
    pub fn init_method(mut self, method: InitMethod) -> Self {
        self.init_method = method;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the number of initialization runs
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the random seed for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: usize) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Fit the k-modes algorithm to the data and return cluster assignments
    pub fn fit<T>(&self, data: ArrayView2<T>) -> Result<KModesResult<T>>
    where
        T: Clone + Eq + Hash + Send + Sync,
    {
        self.validate_input(data)?;

        let mut best_result: Option<KModesResult<T>> = None;
        let mut best_inertia = f64::INFINITY;

        // Run multiple initializations and keep the best result
        let results: Vec<Result<KModesResult<T>>> = if self.should_use_parallel() {
            (0..self.n_init)
                .into_par_iter()
                .map(|i| {
                    let seed = self.random_state.unwrap_or(0) + i as u64;
                    self.fit_single(data, seed)
                })
                .collect()
        } else {
            (0..self.n_init)
                .map(|i| {
                    let seed = self.random_state.unwrap_or(0) + i as u64;
                    self.fit_single(data, seed)
                })
                .collect()
        };

        // Find the best result
        for result in results {
            let result = result?;
            if result.inertia < best_inertia {
                best_inertia = result.inertia;
                best_result = Some(result);
            }
        }

        best_result.ok_or_else(|| Error::convergence_failure("No successful runs"))
    }

    /// Single run of k-modes algorithm
    fn fit_single<T>(&self, data: ArrayView2<T>, seed: u64) -> Result<KModesResult<T>>
    where
        T: Clone + Eq + Hash,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        
        // Initialize centroids
        let mut centroids = initialize_centroids(data, self.n_clusters, self.init_method, &mut rng)?;
        let distance_metric = MatchingDistance;
        
        let mut previous_labels: Option<Array1<usize>> = None;
        let mut n_iter = 0;
        let mut converged = false;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;
            
            // Assign points to closest centroids
            let labels = assign_points_to_centroids(
                data,
                centroids.view(),
                |a, b| distance_metric.distance(a, b),
            )?;

            // Check for convergence
            if let Some(ref prev_labels) = previous_labels {
                if assignments_equal(labels.view(), prev_labels.view()) {
                    converged = true;
                    if self.verbose {
                        println!("K-modes converged after {} iterations", n_iter);
                    }
                    break;
                }
            }

            // Update centroids (compute modes for each cluster)
            let new_centroids = self.update_centroids(data, &labels)?;
            
            // Check if centroids changed significantly
            if let Some(ref _prev_labels) = previous_labels {
                let centroid_change = self.calculate_centroid_change(&centroids, &new_centroids)?;
                if centroid_change < self.tol {
                    converged = true;
                    if self.verbose {
                        println!("K-modes converged (centroid change < tol) after {} iterations", n_iter);
                    }
                    break;
                }
            }

            centroids = new_centroids;
            previous_labels = Some(labels);

            if self.verbose && (iter + 1) % 10 == 0 {
                println!("K-modes iteration {}", iter + 1);
            }
        }

        let final_labels = assign_points_to_centroids(
            data,
            centroids.view(),
            |a, b| distance_metric.distance(a, b),
        )?;

        let inertia = calculate_cost(
            data,
            centroids.view(),
            final_labels.view(),
            |a, b| distance_metric.distance(a, b),
        )?;

        Ok(KModesResult {
            labels: final_labels,
            centroids,
            n_iter,
            inertia,
            converged,
        })
    }

    /// Update centroids by computing the mode of each cluster
    fn update_centroids<T>(&self, data: ArrayView2<T>, labels: &Array1<usize>) -> Result<Array2<T>>
    where
        T: Clone + Eq + Hash,
    {
        let cluster_indices = get_cluster_indices(labels.view(), self.n_clusters);
        let mut new_centroids = Array2::uninit((self.n_clusters, data.ncols()));

        for (cluster_id, indices) in cluster_indices.iter().enumerate() {
            if indices.is_empty() {
                // Handle empty cluster by assigning a random data point as centroid
                let mut rng = StdRng::seed_from_u64(self.random_state.unwrap_or(0) + cluster_id as u64);
                let random_idx = rng.gen_range(0..data.nrows());
                
                for feature_idx in 0..data.ncols() {
                    new_centroids[[cluster_id, feature_idx]].write(data[[random_idx, feature_idx]].clone());
                }
            } else {
                let modes = compute_modes(data, indices)?;
                for (feature_idx, mode) in modes.into_iter().enumerate() {
                    new_centroids[[cluster_id, feature_idx]].write(mode);
                }
            }
        }

        // Safety: we've initialized all elements
        Ok(unsafe { new_centroids.assume_init() })
    }

    /// Calculate the change in centroids between iterations
    fn calculate_centroid_change<T>(&self, old: &Array2<T>, new: &Array2<T>) -> Result<f64>
    where
        T: Clone + PartialEq,
    {
        if old.dim() != new.dim() {
            return Err(Error::computation_error("Centroid dimension mismatch"));
        }

        let mut total_changes = 0;
        let total_elements = old.nrows() * old.ncols();

        for (old_val, new_val) in old.iter().zip(new.iter()) {
            if old_val != new_val {
                total_changes += 1;
            }
        }

        Ok(total_changes as f64 / total_elements as f64)
    }

    /// Validate input parameters and data
    fn validate_input<T>(&self, data: ArrayView2<T>) -> Result<()> {
        validate_parameters(self.n_clusters, self.max_iter, self.tol, self.n_init)?;
        validate_data(data)?;

        if self.n_clusters > data.nrows() {
            return Err(Error::invalid_parameter(
                "Number of clusters cannot exceed number of data points",
            ));
        }

        Ok(())
    }

    /// Determine if parallel processing should be used
    fn should_use_parallel(&self) -> bool {
        match self.n_jobs {
            Some(1) => false,
            Some(_) => true,
            None => self.n_init > 1, // Use parallel by default for multiple inits
        }
    }

    /// Fit the model and predict cluster assignments
    pub fn fit_predict<T>(&self, data: ArrayView2<T>) -> Result<Array1<usize>>
    where
        T: Clone + Eq + Hash + Send + Sync,
    {
        let result = self.fit(data)?;
        Ok(result.labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_kmodes_creation() {
        let kmodes = KModes::new(3);
        assert_eq!(kmodes.n_clusters, 3);
        assert_eq!(kmodes.init_method, InitMethod::Huang);
    }

    #[test]
    fn test_kmodes_builder_pattern() {
        let kmodes = KModes::new(5)
            .init_method(InitMethod::Random)
            .max_iter(50)
            .tolerance(0.001)
            .n_init(5)
            .random_state(42)
            .verbose(true);

        assert_eq!(kmodes.n_clusters, 5);
        assert_eq!(kmodes.init_method, InitMethod::Random);
        assert_eq!(kmodes.max_iter, 50);
        assert_eq!(kmodes.tol, 0.001);
        assert_eq!(kmodes.n_init, 5);
        assert_eq!(kmodes.random_state, Some(42));
        assert!(kmodes.verbose);
    }

    #[test]
    fn test_kmodes_simple_clustering() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec!["A", "X", "A", "X", "B", "Y", "B", "Y", "A", "X", "B", "Y"],
        )
        .unwrap();

        let kmodes = KModes::new(2)
            .random_state(42)
            .n_init(3)
            .max_iter(10);

        let result = kmodes.fit(data.view()).unwrap();
        
        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.centroids.nrows(), 2);
        assert_eq!(result.centroids.ncols(), 2);
        assert!(result.n_iter <= 10);
    }

    #[test]
    fn test_kmodes_convergence() {
        // Create data that should converge quickly
        let data = Array2::from_shape_vec(
            (4, 1),
            vec!["A", "A", "B", "B"],
        ).unwrap();

        let kmodes = KModes::new(2)
            .random_state(42)
            .n_init(1)
            .max_iter(100);

        let result = kmodes.fit(data.view()).unwrap();
        
        assert!(result.converged);
        assert!(result.n_iter < 100);
    }

    #[test]
    fn test_kmodes_fit_predict() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec!["A", "X", "A", "X", "B", "Y", "B", "Y"],
        ).unwrap();

        let kmodes = KModes::new(2).random_state(42);
        let labels = kmodes.fit_predict(data.view()).unwrap();
        
        assert_eq!(labels.len(), 4);
        assert!(labels.iter().all(|&label| label < 2));
    }

    #[test]
    fn test_invalid_parameters() {
        let data = Array2::from_shape_vec((2, 1), vec!["A", "B"]).unwrap();
        
        // Too many clusters
        let kmodes = KModes::new(3);
        assert!(kmodes.fit(data.view()).is_err());
        
        // Zero clusters
        let kmodes = KModes::new(0);
        assert!(kmodes.fit(data.view()).is_err());
    }

    #[test]
    fn test_empty_data() {
        let data = Array2::from_shape_vec((0, 0), Vec::<&str>::new()).unwrap();
        let kmodes = KModes::new(1);
        assert!(kmodes.fit(data.view()).is_err());
    }
}