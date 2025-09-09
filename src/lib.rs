//! # K-modes and K-prototypes Clustering
//!
//! This crate provides implementations of the k-modes and k-prototypes clustering algorithms
//! for categorical and mixed (categorical + numerical) data.
//!
//! ## Features
//!
//! - **K-modes**: Clustering for purely categorical data
//! - **K-prototypes**: Clustering for mixed categorical and numerical data
//! - Multiple initialization strategies (Huang, Cao, random)
//! - Parallel processing support via Rayon
//! - Comprehensive distance metrics for categorical data
//!
//! ## Example
//!
//! ```rust
//! use kategorize::{KModes, InitMethod};
//! use ndarray::Array2;
//!
//! // Create sample categorical data
//! let data = Array2::from_shape_vec((4, 2), vec![
//!     "A", "X", "A", "Y", "B", "X", "B", "Y"
//! ]).unwrap();
//!
//! // Create k-modes clusterer
//! let mut kmodes = KModes::new(2)
//!     .init_method(InitMethod::Huang)
//!     .max_iter(100)
//!     .n_init(10);
//!
//! // Fit the model and get cluster assignments
//! let result = kmodes.fit(data.view()).unwrap();
//! println!("Cluster labels: {:?}", result.labels);
//! ```

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod error;
pub mod kmodes;
pub mod kprototypes;
pub mod distance;
pub mod initialization;
pub mod utils;

pub use error::{Error, Result};
pub use kmodes::{KModes, KModesResult, DistanceMetric};
pub use kprototypes::{KPrototypes, KPrototypesResult, MixedValue};
pub use initialization::InitMethod;
pub use distance::{CategoricalDistance, MatchingDistance, HammingDistance, JaccardDistance, EuclideanDistance};

/// Re-export commonly used types from ndarray
pub use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn basic_functionality() {
        // Basic smoke test to ensure the crate compiles
        let _init_method = InitMethod::Huang;
    }
}