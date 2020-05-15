use nalgebra::{ClosedAdd, ClosedMul, Field};
use num_traits::{Float};
use std::fmt::Debug;

pub trait Number: Float + Field + ClosedAdd + ClosedMul + Debug + Copy + 'static {}

impl Number for f32 {}
impl Number for f64 {}

pub trait Network<N>
where
N: Number,
{
    fn run(&self, input: &[N]) -> Vec<N>;
    fn train(&mut self, input: &[N], output: &[N]);
}
