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

    pub fn update(&mut self, weights: Vec<Value>, bias: Value) {
        self.weights = weights;
        self.bias = bias;
    }

    pub fn forward(&self, inputs: &[Value]) -> Value {
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

    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(inputs)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }

    pub fn update(&mut self, parameters: Vec<Value>) {
        let mut parameters = parameters.into_iter();
        for neuron in &mut self.neurons {
            let weights = parameters.by_ref().take(neuron.weights.len()).collect();
            let bias = parameters.next().unwrap();
            neuron.update(weights, bias);
        }
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

    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        let mut outputs = inputs.to_vec();
        for layer in &self.layers {
            outputs = layer.forward(&outputs);
        }
        outputs
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    pub fn update_parameters(&mut self, parameters: Vec<Value>) {
        let mut parameters = parameters.into_iter();
        for layer in &mut self.layers {
            let mut layer_params = Vec::new();
            for neuron in &layer.neurons {
                let weights: Vec<Value> = parameters.by_ref().take(neuron.weights.len()).collect();
                let bias = parameters.next().unwrap();
                layer_params.extend(weights);
                layer_params.push(bias);
            }
            layer.update(layer_params);
        }
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
        let outputs = n.forward(&x);
        assert_eq!(outputs.len(), 3);
        for o in outputs {
            println!("data: {}", o.data);
        }
    }

    #[test]
    fn test_mlp() {
        let x = [Value::from(2.0_f64), Value::from(3.0_f64), Value::from(-1.0_f64)];
        let mut n = MLP::new(3, &[4, 4, 1]);
        let outputs = n.forward(&x);
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

        for k in 0..20 {
            let ypred = xs.iter().map(|x| n.forward(x)).collect::<Vec<_>>();
            let loss = ypred.iter().zip(ys.iter()).map(|(yp, y)| (&yp[0] - y).sqrt()).fold(Value::default(), |acc, x| acc + x);
            let grad_store = loss.backward();

            let mut parameters = n.parameters();
            for p in parameters.iter_mut() {
                *p = -0.05 * grad_store.0.get(&p.id).unwrap() + p.clone();
            }

            n.update_parameters(parameters);

            println!("{k}, loss: {}", loss.data);
        }
    }
}