use kategorize::{KModes, InitMethod};
use ndarray::Array2;

#[test]
fn test_debug_random_init() {
    let data = Array2::from_shape_vec(
        (4, 2),
        vec!["A", "X", "B", "Y", "A", "X", "B", "Y"]
    ).unwrap();

    let kmodes = KModes::new(2)
        .init_method(InitMethod::Random)
        .random_state(42)
        .n_init(1)
        .max_iter(10);

    match kmodes.fit(data.view()) {
        Ok(result) => {
            println!("Success: {} iterations, inertia: {}", result.n_iter, result.inertia);
            println!("Labels: {:?}", result.labels);
        },
        Err(e) => {
            println!("Error: {:?}", e);
            panic!("Test failed with error: {:?}", e);
        }
    }
}