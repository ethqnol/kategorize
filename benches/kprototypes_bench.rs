use kategorize::{KPrototypes, MixedValue};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array2;
use rand::prelude::*;

fn generate_mixed_data(n_samples: usize, n_categorical: usize, n_numerical: usize) -> (Array2<MixedValue<String>>, Vec<usize>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::with_capacity(n_samples * (n_categorical + n_numerical));
    
    for _ in 0..n_samples {
        // Add categorical features
        for _ in 0..n_categorical {
            let category = rng.gen_range(0..10);
            data.push(MixedValue::Categorical(format!("cat_{}", category)));
        }
        
        // Add numerical features
        for _ in 0..n_numerical {
            let value = rng.gen_range(0.0..100.0);
            data.push(MixedValue::Numerical(value));
        }
    }
    
    let categorical_indices: Vec<usize> = (0..n_categorical).collect();
    let numerical_indices: Vec<usize> = (n_categorical..n_categorical + n_numerical).collect();
    
    let array = Array2::from_shape_vec((n_samples, n_categorical + n_numerical), data).unwrap();
    
    (array, categorical_indices, numerical_indices)
}

fn bench_kprototypes_small(c: &mut Criterion) {
    let (data, cat_indices, num_indices) = generate_mixed_data(100, 3, 2);
    
    let mut group = c.benchmark_group("kprototypes_small");
    
    for &n_clusters in &[2, 5, 8] {
        for &gamma in &[0.5, 1.0, 2.0] {
            group.bench_with_input(
                BenchmarkId::new(format!("k{}_g{}", n_clusters, gamma), ""),
                &(n_clusters, gamma),
                |b, &(k, g)| {
                    let kproto = KPrototypes::new(k, cat_indices.clone(), num_indices.clone())
                        .gamma(g)
                        .random_state(42)
                        .n_init(1)
                        .max_iter(50);
                    
                    b.iter(|| {
                        black_box(kproto.fit(
                            black_box(data.view()),
                            black_box(cat_indices.clone()),
                            black_box(num_indices.clone())
                        ).unwrap())
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_kprototypes_medium(c: &mut Criterion) {
    let (data, cat_indices, num_indices) = generate_mixed_data(500, 5, 5);
    
    let mut group = c.benchmark_group("kprototypes_medium");
    group.sample_size(20);
    
    for &n_clusters in &[5, 10] {
        group.bench_with_input(
            BenchmarkId::new("clusters", n_clusters),
            &n_clusters,
            |b, &k| {
                let kproto = KPrototypes::new(k, cat_indices.clone(), num_indices.clone())
                    .gamma(1.0)
                    .random_state(42)
                    .n_init(1)
                    .max_iter(50);
                
                b.iter(|| {
                    black_box(kproto.fit(
                        black_box(data.view()),
                        black_box(cat_indices.clone()),
                        black_box(num_indices.clone())
                    ).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_gamma_effect(c: &mut Criterion) {
    let (data, cat_indices, num_indices) = generate_mixed_data(200, 4, 3);
    
    let mut group = c.benchmark_group("gamma_effect");
    
    let gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    
    for &gamma in &gammas {
        group.bench_with_input(
            BenchmarkId::from_parameter(gamma),
            &gamma,
            |b, &g| {
                let kproto = KPrototypes::new(5, cat_indices.clone(), num_indices.clone())
                    .gamma(g)
                    .random_state(42)
                    .n_init(1)
                    .max_iter(50);
                
                b.iter(|| {
                    black_box(kproto.fit(
                        black_box(data.view()),
                        black_box(cat_indices.clone()),
                        black_box(num_indices.clone())
                    ).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_mixed_data_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_data_ratios");
    
    let ratios = [
        (8, 2, "mostly_categorical"),
        (5, 5, "balanced"),
        (2, 8, "mostly_numerical"),
    ];
    
    for &(n_cat, n_num, name) in &ratios {
        let (data, cat_indices, num_indices) = generate_mixed_data(300, n_cat, n_num);
        
        group.bench_function(name, |b| {
            let kproto = KPrototypes::new(5, cat_indices.clone(), num_indices.clone())
                .gamma(1.0)
                .random_state(42)
                .n_init(1)
                .max_iter(50);
            
            b.iter(|| {
                black_box(kproto.fit(
                    black_box(data.view()),
                    black_box(cat_indices.clone()),
                    black_box(num_indices.clone())
                ).unwrap())
            });
        });
    }
    
    group.finish();
}

fn bench_kprototypes_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("kprototypes_scaling");
    group.sample_size(10);
    
    let sizes = [
        (100, 3, 2),
        (200, 4, 3),
        (500, 5, 5),
    ];
    
    for &(n_samples, n_cat, n_num) in &sizes {
        let (data, cat_indices, num_indices) = generate_mixed_data(n_samples, n_cat, n_num);
        
        group.bench_with_input(
            BenchmarkId::new("scaling", format!("{}x{}+{}", n_samples, n_cat, n_num)),
            &data,
            |b, data| {
                let kproto = KPrototypes::new(5, cat_indices.clone(), num_indices.clone())
                    .gamma(1.0)
                    .random_state(42)
                    .n_init(1)
                    .max_iter(30);
                
                b.iter(|| {
                    black_box(kproto.fit(
                        black_box(data.view()),
                        black_box(cat_indices.clone()),
                        black_box(num_indices.clone())
                    ).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_kprototypes_small,
    bench_kprototypes_medium,
    bench_gamma_effect,
    bench_mixed_data_ratios,
    bench_kprototypes_scaling
);
criterion_main!(benches);