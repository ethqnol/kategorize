//! K-prototypes clustering algorithm for mixed categorical and numerical data

use crate::distance::{EuclideanDistance, MatchingDistance};
use crate::error::{Error, Result};
use crate::initialization::{InitMethod};
use crate::utils::{
    assignments_equal, get_cluster_indices, validate_data, validate_parameters,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Data type for mixed categorical and numerical features
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MixedValue<T> {
    /// Categorical value
    Categorical(T),
    /// Numerical value
    Numerical(f64),
}

/// K-prototypes clustering algorithm for mixed data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KPrototypes {
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
    /// Number of parallel jobs
    pub n_jobs: Option<usize>,
    /// Enable verbose output
    pub verbose: bool,
    /// Weight for categorical vs numerical features (gamma parameter)
    pub gamma: f64,
}

/// Result of k-prototypes clustering
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KPrototypesResult<T> {
    /// Cluster labels for each data point
    pub labels: Array1<usize>,
    /// Final cluster centroids (prototypes)
    pub centroids: Array2<MixedValue<T>>,
    /// Number of iterations until convergence
    pub n_iter: usize,
    /// Final inertia (total within-cluster distance)
    pub inertia: f64,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Categorical feature indices
    pub categorical_indices: Vec<usize>,
    /// Numerical feature indices  
    pub numerical_indices: Vec<usize>,
}

/// Mixed distance function for k-prototypes
#[derive(Debug, Clone)]
pub struct PrototypesDistance {
    categorical_indices: Vec<usize>,
    numerical_indices: Vec<usize>,
    gamma: f64,
    categorical_distance: MatchingDistance,
    numerical_distance: EuclideanDistance,
}

impl PrototypesDistance {
    /// Create a new prototypes distance metric
    pub fn new(categorical_indices: Vec<usize>, numerical_indices: Vec<usize>, gamma: f64) -> Self {
        Self {
            categorical_indices,
            numerical_indices,
            gamma,
            categorical_distance: MatchingDistance,
            numerical_distance: EuclideanDistance,
        }
    }

    /// Compute distance between two mixed data points
    pub fn distance<T: PartialEq>(
        &self,
        a: ArrayView1<MixedValue<T>>,
        b: ArrayView1<MixedValue<T>>,
    ) -> Result<f64> {
        if a.len() != b.len() {
            return Err(Error::invalid_data("Vectors must have the same length"));
        }

        let mut categorical_distance = 0.0;
        let mut numerical_distance = 0.0;

        // Compute categorical distance
        for &idx in &self.categorical_indices {
            if idx >= a.len() {
                return Err(Error::invalid_data("Categorical index out of bounds"));
            }
            
            match (&a[idx], &b[idx]) {
                (MixedValue::Categorical(x), MixedValue::Categorical(y)) => {
                    if x != y {
                        categorical_distance += 1.0;
                    }
                }
                _ => return Err(Error::invalid_data("Expected categorical values")),
            }
        }

        // Compute numerical distance (squared differences for efficiency)
        for &idx in &self.numerical_indices {
            if idx >= a.len() {
                return Err(Error::invalid_data("Numerical index out of bounds"));
            }

            match (&a[idx], &b[idx]) {
                (MixedValue::Numerical(x), MixedValue::Numerical(y)) => {
                    numerical_distance += (x - y).powi(2);
                }
                _ => return Err(Error::invalid_data("Expected numerical values")),
            }
        }

        // Combine distances with gamma weighting
        Ok(categorical_distance + self.gamma * numerical_distance)
    }

    /// Compute distances between a point and all centroids
    pub fn distances_to_centroids<T: PartialEq>(
        &self,
        point: ArrayView1<MixedValue<T>>,
        centroids: ArrayView2<MixedValue<T>>,
    ) -> Result<Vec<f64>> {
        let mut distances = Vec::with_capacity(centroids.nrows());
        
        for centroid_row in centroids.rows() {
            distances.push(self.distance(point, centroid_row)?);
        }
        
        Ok(distances)
    }
}

impl Default for KPrototypes {
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
            gamma: 1.0,
        }
    }
}

impl KPrototypes {
    /// Create a new k-prototypes clusterer
    pub fn new(
        n_clusters: usize,
        categorical_indices: Vec<usize>,
        numerical_indices: Vec<usize>,
    ) -> Self {
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

    /// Set the gamma parameter (weight for numerical vs categorical features)
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Fit the k-prototypes algorithm to mixed data
    pub fn fit<T>(
        &self,
        data: ArrayView2<MixedValue<T>>,
        categorical_indices: Vec<usize>,
        numerical_indices: Vec<usize>,
    ) -> Result<KPrototypesResult<T>>
    where
        T: Clone + Eq + Hash + Send + Sync,
    {
        self.validate_input(data, &categorical_indices, &numerical_indices)?;

        let mut best_result: Option<KPrototypesResult<T>> = None;
        let mut best_inertia = f64::INFINITY;

        // Run multiple initializations and keep the best result
        let results: Vec<Result<KPrototypesResult<T>>> = if self.should_use_parallel() {
            (0..self.n_init)
                .into_par_iter()
                .map(|i| {
                    let seed = self.random_state.unwrap_or(0) + i as u64;
                    self.fit_single(data, &categorical_indices, &numerical_indices, seed)
                })
                .collect()
        } else {
            (0..self.n_init)
                .map(|i| {
                    let seed = self.random_state.unwrap_or(0) + i as u64;
                    self.fit_single(data, &categorical_indices, &numerical_indices, seed)
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

    /// Single run of k-prototypes algorithm
    fn fit_single<T>(
        &self,
        data: ArrayView2<MixedValue<T>>,
        categorical_indices: &[usize],
        numerical_indices: &[usize],
        seed: u64,
    ) -> Result<KPrototypesResult<T>>
    where
        T: Clone + Eq + Hash,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        
        // Initialize centroids
        let mut centroids = self.initialize_centroids(data, &mut rng)?;
        let distance_metric = PrototypesDistance::new(
            categorical_indices.to_vec(),
            numerical_indices.to_vec(),
            self.gamma,
        );

        let mut previous_labels: Option<Array1<usize>> = None;
        let mut n_iter = 0;
        let mut converged = false;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Assign points to closest centroids
            let labels = self.assign_points_to_centroids(data, centroids.view(), &distance_metric)?;

            // Check for convergence
            if let Some(ref prev_labels) = previous_labels {
                if assignments_equal(labels.view(), prev_labels.view()) {
                    converged = true;
                    if self.verbose {
                        println!("K-prototypes converged after {} iterations", n_iter);
                    }
                    break;
                }
            }

            // Update centroids
            let new_centroids = self.update_centroids(data, &labels, categorical_indices, numerical_indices)?;

            centroids = new_centroids;
            previous_labels = Some(labels);

            if self.verbose && (iter + 1) % 10 == 0 {
                println!("K-prototypes iteration {}", iter + 1);
            }
        }

        let final_labels = self.assign_points_to_centroids(data, centroids.view(), &distance_metric)?;
        let inertia = self.calculate_inertia(data, centroids.view(), final_labels.view(), &distance_metric)?;

        Ok(KPrototypesResult {
            labels: final_labels,
            centroids,
            n_iter,
            inertia,
            converged,
            categorical_indices: categorical_indices.to_vec(),
            numerical_indices: numerical_indices.to_vec(),
        })
    }

    /// Initialize centroids for mixed data
    fn initialize_centroids<T, R>(
        &self,
        data: ArrayView2<MixedValue<T>>,
        rng: &mut R,
    ) -> Result<Array2<MixedValue<T>>>
    where
        T: Clone,
        R: Rng,
    {
        // For now, use random initialization
        // In a full implementation, we'd adapt the initialization methods for mixed data
        let mut selected_indices = std::collections::HashSet::new();
        let n_points = data.nrows();

        while selected_indices.len() < self.n_clusters {
            let idx = rng.gen_range(0..n_points);
            selected_indices.insert(idx);
        }

        let indices: Vec<_> = selected_indices.into_iter().collect();
        let mut centroids = Array2::uninit((self.n_clusters, data.ncols()));

        for (i, &data_idx) in indices.iter().enumerate() {
            for j in 0..data.ncols() {
                centroids[[i, j]].write(data[[data_idx, j]].clone());
            }
        }

        // Safety: we've initialized all elements
        Ok(unsafe { centroids.assume_init() })
    }

    /// Assign points to closest centroids using mixed distance
    fn assign_points_to_centroids<T>(
        &self,
        data: ArrayView2<MixedValue<T>>,
        centroids: ArrayView2<MixedValue<T>>,
        distance_metric: &PrototypesDistance,
    ) -> Result<Array1<usize>>
    where
        T: PartialEq,
    {
        let mut assignments = Array1::zeros(data.nrows());

        for (i, point) in data.rows().into_iter().enumerate() {
            let distances = distance_metric.distances_to_centroids(point, centroids)?;
            
            let closest = distances
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .ok_or_else(|| Error::computation_error("No centroids found"))?;
            
            assignments[i] = closest;
        }

        Ok(assignments)
    }

    /// Update centroids for mixed data
    fn update_centroids<T>(
        &self,
        data: ArrayView2<MixedValue<T>>,
        labels: &Array1<usize>,
        categorical_indices: &[usize],
        numerical_indices: &[usize],
    ) -> Result<Array2<MixedValue<T>>>
    where
        T: Clone + Eq + Hash,
    {
        let cluster_indices = get_cluster_indices(labels.view(), self.n_clusters);
        let mut new_centroids = Array2::uninit((self.n_clusters, data.ncols()));

        for (cluster_id, indices) in cluster_indices.iter().enumerate() {
            if indices.is_empty() {
                return Err(Error::computation_error(format!(
                    "Empty cluster {} during centroid update",
                    cluster_id
                )));
            }

            // Update each feature
            for feature_idx in 0..data.ncols() {
                let new_value = if categorical_indices.contains(&feature_idx) {
                    // Categorical feature: compute mode
                    let values: Vec<_> = indices
                        .iter()
                        .filter_map(|&row_idx| {
                            match &data[[row_idx, feature_idx]] {
                                MixedValue::Categorical(val) => Some(val.clone()),
                                _ => None,
                            }
                        })
                        .collect();

                    let mode = self.compute_categorical_mode(&values)?;
                    MixedValue::Categorical(mode)
                } else if numerical_indices.contains(&feature_idx) {
                    // Numerical feature: compute mean
                    let values: Vec<f64> = indices
                        .iter()
                        .filter_map(|&row_idx| {
                            match &data[[row_idx, feature_idx]] {
                                MixedValue::Numerical(val) => Some(*val),
                                _ => None,
                            }
                        })
                        .collect();

                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    MixedValue::Numerical(mean)
                } else {
                    return Err(Error::invalid_data("Feature index not in categorical or numerical indices"));
                };

                new_centroids[[cluster_id, feature_idx]].write(new_value);
            }
        }

        // Safety: we've initialized all elements
        Ok(unsafe { new_centroids.assume_init() })
    }

    /// Compute mode for categorical values
    fn compute_categorical_mode<T: Clone + Eq + Hash>(&self, values: &[T]) -> Result<T> {
        if values.is_empty() {
            return Err(Error::computation_error("Cannot compute mode of empty values"));
        }

        let mut counts = HashMap::new();
        for value in values {
            *counts.entry(value.clone()).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(value, _)| value)
            .ok_or_else(|| Error::computation_error("Unable to compute mode"))
    }

    /// Calculate inertia for mixed data
    fn calculate_inertia<T>(
        &self,
        data: ArrayView2<MixedValue<T>>,
        centroids: ArrayView2<MixedValue<T>>,
        assignments: ArrayView1<usize>,
        distance_metric: &PrototypesDistance,
    ) -> Result<f64>
    where
        T: PartialEq,
    {
        let mut total_cost = 0.0;

        for (i, point) in data.rows().into_iter().enumerate() {
            let cluster_id = assignments[i];
            if cluster_id >= centroids.nrows() {
                return Err(Error::invalid_data("Invalid cluster assignment"));
            }

            let centroid = centroids.row(cluster_id);
            total_cost += distance_metric.distance(point, centroid)?;
        }

        Ok(total_cost)
    }

    /// Validate input parameters and data
    fn validate_input<T>(
        &self,
        data: ArrayView2<MixedValue<T>>,
        categorical_indices: &[usize],
        numerical_indices: &[usize],
    ) -> Result<()> {
        validate_parameters(self.n_clusters, self.max_iter, self.tol, self.n_init)?;
        validate_data(data)?;

        if self.n_clusters > data.nrows() {
            return Err(Error::invalid_parameter(
                "Number of clusters cannot exceed number of data points",
            ));
        }

        // Check that all indices are valid
        let max_index = data.ncols();
        for &idx in categorical_indices.iter().chain(numerical_indices.iter()) {
            if idx >= max_index {
                return Err(Error::invalid_parameter("Feature index out of bounds"));
            }
        }

        // Check that indices don't overlap
        let mut all_indices = categorical_indices.to_vec();
        all_indices.extend_from_slice(numerical_indices);
        all_indices.sort_unstable();
        
        for window in all_indices.windows(2) {
            if window[0] == window[1] {
                return Err(Error::invalid_parameter("Duplicate feature indices"));
            }
        }

        if self.gamma < 0.0 {
            return Err(Error::invalid_parameter("Gamma must be non-negative"));
        }

        Ok(())
    }

    /// Determine if parallel processing should be used
    fn should_use_parallel(&self) -> bool {
        match self.n_jobs {
            Some(1) => false,
            Some(_) => true,
            None => self.n_init > 1,
        }
    }

    /// Fit the model and predict cluster assignments
    pub fn fit_predict<T>(
        &self,
        data: ArrayView2<MixedValue<T>>,
        categorical_indices: Vec<usize>,
        numerical_indices: Vec<usize>,
    ) -> Result<Array1<usize>>
    where
        T: Clone + Eq + Hash + Send + Sync,
    {
        let result = self.fit(data, categorical_indices, numerical_indices)?;
        Ok(result.labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_mixed_value_equality() {
        let cat1 = MixedValue::Categorical("A");
        let cat2 = MixedValue::Categorical("A");
        let cat3 = MixedValue::Categorical("B");
        let num1 = MixedValue::Numerical(1.0);
        let num2 = MixedValue::Numerical(1.0);

        assert_eq!(cat1, cat2);
        assert_ne!(cat1, cat3);
        assert_eq!(num1, num2);
        assert_ne!(cat1, num1);
    }

    #[test]
    fn test_prototypes_distance() {
        let distance = PrototypesDistance::new(vec![0], vec![1], 1.0);
        
        let a = ndarray::arr1(&[MixedValue::Categorical("A"), MixedValue::Numerical(1.0)]);
        let b = ndarray::arr1(&[MixedValue::Categorical("B"), MixedValue::Numerical(2.0)]);
        
        let result = distance.distance(a.view(), b.view()).unwrap();
        // Should be 1.0 (categorical mismatch) + 1.0 * (1.0)^2 (numerical difference squared)
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_kprototypes_creation() {
        let kproto = KPrototypes::new(3, vec![0, 1], vec![2, 3]);
        assert_eq!(kproto.n_clusters, 3);
        assert_eq!(kproto.gamma, 1.0);
    }

    #[test]
    fn test_kprototypes_simple_clustering() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                MixedValue::Categorical("A"), MixedValue::Categorical("X"), MixedValue::Numerical(1.0),
                MixedValue::Categorical("A"), MixedValue::Categorical("X"), MixedValue::Numerical(2.0),
                MixedValue::Categorical("B"), MixedValue::Categorical("Y"), MixedValue::Numerical(10.0),
                MixedValue::Categorical("B"), MixedValue::Categorical("Y"), MixedValue::Numerical(11.0),
            ],
        ).unwrap();

        let kproto = KPrototypes::new(2, vec![0, 1], vec![2])
            .random_state(42)
            .n_init(3)
            .max_iter(10);

        let result = kproto.fit(data.view(), vec![0, 1], vec![2]).unwrap();
        
        assert_eq!(result.labels.len(), 4);
        assert_eq!(result.centroids.nrows(), 2);
        assert_eq!(result.centroids.ncols(), 3);
    }

    #[test]
    fn test_invalid_feature_indices() {
        let data = Array2::from_shape_vec(
            (2, 2),
            vec![
                MixedValue::Categorical("A"), MixedValue::Numerical(1.0),
                MixedValue::Categorical("B"), MixedValue::Numerical(2.0),
            ],
        ).unwrap();

        let kproto = KPrototypes::new(2, vec![0], vec![1]);
        
        // Test invalid categorical index
        assert!(kproto.fit(data.view(), vec![2], vec![1]).is_err());
        
        // Test duplicate indices
        assert!(kproto.fit(data.view(), vec![0], vec![0]).is_err());
    }
}