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
        let mut network = FeedForward::<f32>::new(&[2, 3, 8, 4, 2]);
        println!("{:?} {:?}", network.run(&[1.0, 0.0]), network.run(&[0.0, 1.0]));
        for _ in 0..1000 {
            network.train(&[1.0, 0.0], &[1.0, 0.0]);
            network.train(&[0.0, 1.0], &[0.0, 1.0]);
            println!("{:?} {:?}", network.run(&[1.0, 0.0]), network.run(&[0.0, 1.0]));
        }
    }
}
