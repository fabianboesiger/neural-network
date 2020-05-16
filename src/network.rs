use num_traits::{Float, NumOps, NumAssignOps};
use std::fmt::Debug;
use ndarray::{Array1, Array2};

pub type Vector<N> = Array1<N>;
pub type Matrix<N> = Array2<N>;

pub trait Number: Float + NumOps + NumAssignOps + Debug + Copy + Send + Sync + 'static {}

impl Number for f32 {}
impl Number for f64 {}

pub trait Network<N>
where
N: Number,
{
    /// Runs the network on an input and returns the output.
    fn run(&self, input: &[N]) -> Vec<N>;

    /// Trains the network until the target error is reached.
    fn train(&mut self, target: N, data: Vec<(Vec<N>, Vec<N>)>);    
  
}
