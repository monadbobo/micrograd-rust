use std::fmt::Display;
use rand::Rng;
use crate::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct Neuron {
    pub weights: Vec<Value>,
    pub bias: Value,
    pub non_lin: bool,
}

impl Neuron {
    pub fn new(nin: usize, non_lin: bool) -> Neuron {
        Neuron {
            weights: (0..nin).map(|_| Value::new(rand::thread_rng().gen_range(-1.0..=1.0))).collect::<Vec<_>>(),
            bias: Value::default(),
            non_lin,
        }
    }

    pub fn exec(&self, inputs: &[Value]) -> Value {
        let act = self.weights.iter().zip(inputs.iter()).fold(self.bias.clone(), |acc, (w, i)| acc + w * i);
        if self.non_lin {
            act.tanh()
        } else {
            act
        }
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.weights.iter().cloned().chain(std::iter::once(self.bias.clone())).collect()
    }
}

// impl Display for Neuron {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {}
// }

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Layer {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin, true)).collect::<Vec<_>>(),
        }
    }

    pub fn exec(&self, inputs: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.exec(inputs)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize]) -> MLP {
        let mut layers = Vec::new();
        let mut n = nin;
        for &nout in nouts {
            layers.push(Layer::new(n, nout));
            n = nout;
        }
        MLP { layers }
    }

    pub fn exec(&self, inputs: &[Value]) -> Vec<Value> {
        let mut outputs = inputs.to_vec();
        for layer in &self.layers {
            outputs = layer.exec(&outputs);
        }
        outputs
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

#[cfg(test)]
mod test {
    use crate::nn::{Layer, MLP};
    use crate::Value;

    #[test]
    fn test_layer() {
        let x = [Value::from(2.0_f64), Value::from(3.0_f64)];
        let n = Layer::new(2, 3);
        let outputs = n.exec(&x);
        assert_eq!(outputs.len(), 3);
        for o in outputs {
            println!("data: {}", o.data);
        }
    }

    #[test]
    fn test_mlp() {
        let x = [Value::from(2.0_f64), Value::from(3.0_f64), Value::from(-1.0_f64)];
        let n = MLP::new(3, &[4, 4, 1]);
        let outputs = n.exec(&x);
        assert_eq!(outputs.len(), 1);
        println!("data: {}", outputs[0].data);
        for o in n.parameters() {
            println!("parameters: {}", o.data);
        }
        assert_eq!(n.parameters().len(), 41);

        let xs = [[Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
            [Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
            [Value::new(0.5), Value::new(1.0), Value::new(1.0)],
            [Value::new(1.0), Value::new(1.0), Value::new(-1.0)]];

        let ys = [Value::new(1.0), Value::new(-1.0), Value::new(-1.0), Value::new(1.0)];
        let ypred = xs.iter().map(|x| n.exec(x)[0].clone()).collect::<Vec<_>>();
        for (y, ypred) in ys.iter().zip(ypred.iter()) {
            println!("y: {}, ypred: {}", y.data, ypred.data);
        }
        let mut loss = Value::new(0.0);
        for (y, ypred) in ys.iter().zip(ypred.iter()) {
            loss = loss + (y - ypred).pow(&Value::new(2.0));
        }
        println!("loss: {}", loss.data);
    }
}