mod feed_forward;
mod network;

pub use feed_forward::*;
pub use network::*;

#[cfg(test)]
mod tests {
    use super::FeedForward;
    use super::Network;

    #[test]
    fn it_works() {
        let mut network = FeedForward::<f32>::new(&[2, 10, 20, 30, 20, 10, 4]);
        
        let data = vec![
            (vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]),
            (vec![0.0, 1.0], vec![0.0, 1.0, 0.0, 0.0]),
            (vec![1.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]),
            (vec![1.0, 1.0], vec![0.0, 0.0, 0.0, 1.0])
        ];

        network.train(0.001, data);
    }
}
