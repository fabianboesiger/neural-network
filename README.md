 # Neural Networks in Rust
 
 This crate provides a collection of neural networks implemented in rust using [`Rayon`](https://github.com/rayon-rs/rayon) for parallelism.
 
 ## How to Use
 
 Add the following to your `Cargo.toml`:
 
 ```toml
 [dependencies]
 neural_network = { git = "https://github.com/fabianboesiger/neural-network" }
 ```
 
 ## Network Types
 
 ### Feedforward Network
 
 A feedforward neural network using backpropagation as a learning strategy.
 
 ```rust
 use neural_network::FeedForward;
 use neural_network::Network;

fn main() {
    // Create a neural network using the `f32` data type with the specified layer sizes.
    let mut network = FeedForward::<f32>::new(&[2, 10, 20, 30, 20, 10, 4]);
    
    // Some data to train on.
    // The data is a vector of (input, output) tuples.
    let data = vec![
        (vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]),
        (vec![0.0, 1.0], vec![0.0, 1.0, 0.0, 0.0]),
        (vec![1.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]),
        (vec![1.0, 1.0], vec![0.0, 0.0, 0.0, 1.0]),
    ];
    
    // Train the network, 0.01 is the desired error.
    // The training process is parallelized.
    network.train(0.01, data);
    
    // Print an estimation for the specified input.
    println!("{:?}", network.run(&[0.5, 1.0]));
}
```

### Jordan Network

Not yet implemented.
