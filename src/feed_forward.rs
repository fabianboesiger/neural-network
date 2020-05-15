use crate::{Network, Number};
use nalgebra::{DMatrix, DVector};
use rand::distributions::{Distribution, Standard};

const LEARNING_RATE: f32 = 0.1;

#[derive(Debug)]
pub struct FeedForward<'a, N>
where
    N: Number,
{
    layers: &'a [usize],
    axons: Vec<DMatrix<N>>,
}

impl<'a, N> FeedForward<'a, N>
where
    N: Number,
    Standard: Distribution<N>,
{
    pub fn new(layers: &[usize]) -> FeedForward<N> {
        FeedForward {
            layers,
            axons: {
                let mut output = Vec::with_capacity(layers.len() - 1);
                for (i, j) in layers.iter().zip(layers.iter().skip(1)) {
                    let mut matrix = DMatrix::zeros(*j + 1, *i + 1);
                    for (x, y) in matrix
                        .slice_mut((0, 0), (*j, *i + 1))
                        .iter_mut()
                        .zip(DMatrix::new_random(*j, *i + 1).into_iter())
                    {
                        *x = *y * N::from(2).unwrap() - N::from(1).unwrap();
                    }
                    *matrix.index_mut((*j, *i)) = N::one();
                    output.push(matrix);
                }
                output
            },
        }
    }
    
    fn vector_from_slice(slice: &[N]) -> DVector<N> {
        let mut output = DVector::zeros(slice.len());
        output.copy_from_slice(slice);
        output
    }

    fn run_full(&self, input: &[N]) -> Vec<DVector<N>> {
        let mut output = Vec::with_capacity(self.axons.len() + 1);

        output.push(FeedForward::vector_from_slice(input).map(N::tanh).push(N::one()));

        for layer in &self.axons {
            output.push((layer * output.last().unwrap()).map(N::tanh));
        }

        output
    }
}

impl<N> Network<N> for FeedForward<'_, N>
where
    N: Number,
    Standard: Distribution<N>
{
    fn run(&self, input: &[N]) -> Vec<N> {
        let output = self.run_full(input).pop().unwrap();
        let output_size = output.nrows() - 1;
        output.remove_row(output_size).map(N::atanh).data.into()
    }

    fn train(&mut self, input: &[N], output: &[N]) {
        let output = FeedForward::vector_from_slice(output).map(N::tanh).push(N::one());

        let mut result = self.run_full(input);
        
        let mut error = DVector::zeros(0);
        for i in (0..self.axons.len()).rev() {
            let next = result.pop().unwrap();

            let mut new_error = next.map(|x| N::one() - x * x);
            if i == self.axons.len() - 1 {
                new_error.zip_apply(&(&next - &output), |a, b| a * b);
            } else {
                new_error.zip_apply(&(&self.axons[i + 1].transpose() * &error), |a, b| a * b);
            }
            error = new_error;
        
            let rows = self.axons[i].nrows();
            for (j, axon) in 
                self.axons[i]
                .iter_mut()
                .enumerate()
            {
                let col = j / rows;
                let row = j % rows;

                if row < error.len() - 1 {
                    debug_assert!(col < result.last().unwrap().len());
                    *axon -= N::from(LEARNING_RATE).unwrap() * error[row] * result.last().unwrap()[col];
                }
            }
        }
    }
}
