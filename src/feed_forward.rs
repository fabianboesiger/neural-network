use crate::{Network, Number, Matrix, Vector};
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;
use ndarray::{s, arr1, Axis, stack};

const LEARNING_RATE: f32 = 0.001;

/// A feed forward network using backpropagation.
#[derive(Debug)]
pub struct FeedForward<'a, N>
where
    N: Number,
{
    layers: &'a [usize],
    axons: Vec<Matrix<N>>,
}

impl<'a, N> FeedForward<'a, N>
where
    N: Number,
    Standard: Distribution<N>,
{
    /// Create a new feed forward network with the specified layer sizes.
    pub fn new(layers: &[usize]) -> FeedForward<N> {
        FeedForward {
            layers,
            axons: {
                let mut output = Vec::with_capacity(layers.len() - 1);
                for (a, b) in layers.iter().zip(layers.iter().skip(1)) {
                    let matrix = Matrix::from_shape_fn((*b + 1, *a + 1), |(i, j)| {
                        if i < *b {
                            rand::random::<N>() * (N::one() + N::one()) - N::one()
                        } else {
                            if j < *a {
                                N::zero()
                            } else {
                                N::one()
                            }
                        }
                    });
                    /*
                    for (x, y) in matrix
                        .slice_mut(s![0..*j, 0..(*i + 1)])
                        .iter_mut()
                        .zip(Matrix::new_random(*j, *i + 1).into_iter())
                    {
                        *x = *y * N::from(2).unwrap() - N::from(1).unwrap();
                    }
                    *matrix.index_mut((*j, *i)) = N::one();
                    */
                    output.push(matrix);
                    
                }
                output
            },
        }
    }

    fn error(&self, input: &[N], output: &[N]) -> N {
        let result = Vector::from(self.run(input));
        let output = arr1(output);
        let error = (&result - &output).mapv(|x| x * x).sum();
        error
    }

    fn run_full(&self, input: &[N]) -> Vec<Vector<N>> {
        let mut output = Vec::with_capacity(self.axons.len() + 1);
        //output.push(vector_from_slice(input).map(N::tanh).push(N::one()));
        let input: Vector<N> = stack![Axis(0), arr1(input).mapv(N::tanh), arr1(&[N::one()])];

        output.push(input);

        for layer in &self.axons {
            output.push(layer.dot(output.last().unwrap()).mapv(N::tanh));
        }

        output
    }

    fn train_some(&self, input: &[N], output: &[N]) -> Vec<Matrix<N>> {
        //let output = vector_from_slice(output).map(N::tanh).push(N::one());
        let output = stack![Axis(0), arr1(output).mapv(N::tanh), arr1(&[N::one()])];
        let mut masks = Vec::new();
        let mut result = self.run_full(input);
        let mut error = Vector::zeros(0);

        for i in (0..self.axons.len()).rev() {
            let next = result.pop().unwrap();

            let mut new_error = next.mapv(|x| N::one() - x * x);
            if i == self.axons.len() - 1 {
                new_error *= &(&next - &output);
            } else {
                new_error *= &self.axons[i + 1].t().dot(&error);
            }
            error = new_error;
            
            /*
            let cols = self.axons[i].ncols();
            for (j, axon) in 
                self.axons[i]
                .iter_mut()
                .enumerate()
            {
                let col = j % cols;
                let row = j / cols;

                //println!("{} {} {} \t{:?}", i, row, col, *axon);

                if row < error.len() - 1 {
                    debug_assert!(col < result.last().unwrap().len());
                    *axon -= ;
                }
            }
            */
            let mask = Matrix::from_shape_fn(self.axons[i].raw_dim(), |(i, j)| {
                -N::from(LEARNING_RATE).unwrap() * error[i] * result.last().unwrap()[j]
            });
            masks.push(mask);
        }

        masks
    }
}

impl<N> Network<N> for FeedForward<'_, N>
where
    N: Number,
    Standard: Distribution<N>
{
    fn run(&self, input: &[N]) -> Vec<N> {
        let output = self.run_full(input).pop().unwrap();
        
        let output_size = output.dim() - 1;
        /*
        output.remove_row(output_size).map(N::atanh).data.into()
        */
        output.slice_move(s![0..output_size]).mapv(N::atanh).to_vec()
    }

    fn train(&mut self, target: N, data: Vec<(Vec<N>, Vec<N>)>) {
        loop {
            /*
            for (input, output) in &data {
                self.train(&input, &output)
            }
            */
            let masks: Vec<Vec<Matrix<N>>> = data
                .par_iter()
                .map(|(input, output)| {
                    self.train_some(input, output)
                })
                .collect();

            for masks in masks {
                for (axon, mask) in self.axons.iter_mut().zip(masks.iter().rev()) {
                    *axon += mask;
                }
            }

            let mut total_error = N::zero();
            for (input, output) in &data {
                total_error += self.error(&input, &output);
            }
            total_error /= N::from(data.len()).unwrap();
            println!("{:?}", total_error);
            if total_error <= target {
                break;
            }
        }
    }
}
