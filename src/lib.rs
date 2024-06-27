use std::cell::{RefCell};
use std::collections::{HashMap};
use std::fmt::{Debug};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Add { lhs: Value, rhs: Value },
    Sub { lhs: Value, rhs: Value },
    Mul { lhs: Value, rhs: Value },
    Div { lhs: Value, rhs: Value },
    Tanh { x: Value },
    Exp { x: Value },
    Pow { lhs: Value, rhs: Value },
}

impl Op {
    pub fn backward(&self, out: &Value) -> (f64, f64) {
        match self {
            Op::Add { .. } => (1.0 * out.as_deref().grad, 1.0 * out.as_deref().grad),
            Op::Sub { .. } => unimplemented!(),
            Op::Mul { lhs, rhs } => (lhs.as_deref().data * out.as_deref().grad, rhs.as_deref().data * out.as_deref().grad),
            Op::Div { .. } => unimplemented!(),
            Op::Tanh { .. } => ((1.0 - out.as_deref().data.powi(2)) * out.as_deref().grad, 0.0),
            Op::Exp { x } => (x.as_deref().data.exp() * out.as_deref().grad, 0.0),
            Op::Pow { lhs, rhs } => (lhs.as_deref().data.powf(rhs.as_deref().data - 1.0) * rhs.as_deref().data * out.as_deref().grad, 0.0),
        }
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, Op::Add { .. } | Op::Sub { .. } | Op::Mul { .. } | Op::Div { .. } | Op::Pow { .. })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Value_ {
    pub data: f64,
    pub op: Option<Op>,
    pub label: String,
    pub grad: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Value(Rc<RefCell<Value_>>);

impl Eq for Value {}

// impl Hash for Value {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         self.0.label.hash(state);
//     }
// }

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value(Rc::new(RefCell::new(Value_ {
            data: value,
            op: None,
            label: "".to_string(),
            grad: 0.0,
        })))
    }
}

impl Value {
    fn as_deref(&self) -> impl Deref<Target=Value_> + '_ {
        self.0.borrow()
    }

    fn as_deref_mut(&self) -> impl DerefMut<Target=Value_> + '_ {
        self.0.borrow_mut()
    }

    pub fn tanh(&self) -> Value {
        let x = self.as_deref().data;
        let t = ((x * 2.0).exp() - 1.0) / ((x * 2.0).exp() + 1.0);
        Value(Rc::new(RefCell::new(Value_ {
            data: t,
            op: Some(Op::Tanh { x: self.clone() }),
            label: "".to_string(),
            grad: 0.0,
        })))
    }

    pub fn backward(&mut self) {
        fn build_topo(value: &Value, visited: &mut HashMap<String, bool>, topo: &mut Vec<Value>) {
            if visited.contains_key(&value.as_deref().label) {
                return;
            }
            visited.insert(value.as_deref().label.clone(), true);
            if let Some(op) = &value.0.borrow().op {
                match op {
                    Op::Add { lhs, rhs } | Op::Sub { lhs, rhs } | Op::Mul { lhs, rhs } | Op::Div { lhs, rhs } | Op::Pow { lhs, rhs } => {
                        build_topo(lhs, visited, topo);
                        build_topo(rhs, visited, topo);
                    }
                    Op::Tanh { x } | Op::Exp { x } => {
                        build_topo(x, visited, topo);
                    }
                }
            }
            topo.push(value.clone());
        }
        let mut topo = Vec::new();
        let mut visted = HashMap::new();
        build_topo(self, &mut visted, &mut topo);

        self.as_deref_mut().grad = 1.0;
        for v in topo.iter().rev() {
            if let Some(op) = &v.as_deref().op {
                match op {
                    Op::Add { lhs, rhs } => {
                        lhs.as_deref_mut().grad += v.as_deref().grad;
                        rhs.as_deref_mut().grad += v.as_deref().grad;
                    }
                    Op::Sub { lhs, rhs } => {
                        lhs.as_deref_mut().grad += v.as_deref().grad;
                        rhs.as_deref_mut().grad -= v.as_deref().grad;
                    }
                    Op::Mul { lhs, rhs } => {
                        lhs.as_deref_mut().grad += rhs.as_deref().data * v.as_deref().grad;
                        rhs.as_deref_mut().grad += lhs.as_deref().data * v.as_deref().grad;
                    }
                    Op::Div { lhs, rhs } => {
                        lhs.as_deref_mut().grad += v.as_deref().grad / rhs.as_deref().data;
                        rhs.as_deref_mut().grad += -v.as_deref().grad * lhs.as_deref().data / (rhs.as_deref().data * rhs.as_deref().data);
                    }
                    Op::Pow { lhs, rhs } => {
                        lhs.as_deref_mut().grad += rhs.as_deref().data * lhs.as_deref().data.powf(rhs.as_deref().data - 1.0) * v.as_deref().grad;
                        rhs.as_deref_mut().grad += lhs.as_deref().data.powf(rhs.as_deref().data) * v.as_deref().grad;
                    }
                    Op::Tanh { x } => {
                        x.as_deref_mut().grad += (1.0 - v.as_deref().data.powi(2)) * v.as_deref().grad;
                    }
                    Op::Exp { x } => {
                        x.as_deref_mut().grad += v.as_deref().data.exp() * v.as_deref().grad;
                    }
                }
            }
        }
    }

    pub fn add(&self, other: Value) -> Value {
        Value(Rc::new(RefCell::new(Value_ {
            data: self.as_deref().data + other.as_deref().data,
            op: Some(Op::Add { lhs: self.clone(), rhs: other.clone() }),
            label: "".to_string(),
            grad: 0.0,
        })))
    }

    pub fn mul(&self, other: Value) -> Value {
        Value(Rc::new(RefCell::new(Value_ {
            data: self.as_deref().data * other.as_deref().data,
            op: Some(Op::Mul { lhs: self.clone(), rhs: other.clone() }),
            label: "".to_string(),
            grad: 0.0,
        })))
    }

    pub fn exp(&self) -> Value {
        Value(Rc::new(RefCell::new(Value_ {
            data: self.as_deref().data.exp(),
            op: Some(Op::Exp { x: self.clone() }),
            label: "".to_string(),
            grad: 0.0,
        })))
    }

    pub fn pow(&self, other: Value) -> Value {
        Value(Rc::new(RefCell::new(Value_ {
            data: self.as_deref().data.powf(other.as_deref().data),
            op: Some(Op::Pow { lhs: self.clone(), rhs: other.clone() }),
            label: "".to_string(),
            grad: 0.0,
        })))
    }

    pub fn div(&self, other: Value) -> Value {
        self.mul(other.pow(Value::from(-1.0)))
    }

    pub fn neg(&self) -> Value {
        self.mul(Value::from(-1.0))
    }

    pub fn sub(&self, other: Value) -> Value {
        self.add(other.neg())
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use super::*;

    //#[test]
    // fn test_value() {
    //     let a = Value { data: 2.0, prev: vec![], op: None, label: "a".to_string(), grad: 0.0 };
    //     let b = Value { data: -3.0, prev: vec![], op: None, label: "b".to_string(), grad: 0.0 };
    //     let c = Value { data: 10.0, prev: vec![], op: None, label: "c".to_string(), grad: 0.0 };
    //     let mut e = a * b;
    //     e.label = "e".to_string();
    //     let mut d = e + c;
    //     d.label = "d".to_string();
    //     let mut f = Value { data: -2.0, prev: vec![], op: None, label: "f".to_string(), grad: 0.0 };
    //     f.grad = 4.0;
    //     let mut l = d * f;
    //     l.label = "l".to_string();
    //     l.grad = 1.0;
    //
    //     println!("{}", l);
    // }
    //
    // #[test]
    // fn test_tanh() {
    //     let x1 = Rc::new(RefCell::new(Value { data: 2.0, prev: vec![], op: None, label: "x1".to_string(), grad: 0.0 }));
    //     let x2 = Rc::new(RefCell::new(Value { data: 0.0, prev: vec![], op: None, label: "x2".to_string(), grad: 0.0 }));
    //
    //     let w1 = Rc::new(RefCell::new(Value { data: -3.0, prev: vec![], op: None, label: "w1".to_string(), grad: 0.0 }));
    //     let w2 = Rc::new(RefCell::new(Value { data: 1.0, prev: vec![], op: None, label: "w2".to_string(), grad: 0.0 }));
    //
    //     let b = Rc::new(RefCell::new(Value { data: 6.8813735870195432, prev: vec![], op: None, label: "b".to_string(), grad: 0.0 }));
    //
    //     let mut x1w1 = Value::mul(x1.clone(), w1.clone());
    //     x1w1.label = "x1w1".to_string();
    //     let mut x2w2 = Value::mul(x2, w2);
    //     x2w2.label = "x2w2".to_string();
    //
    //     let mut x1w1x2w2 = Value::add(Rc::new(RefCell::new(x1w1)), Rc::new(RefCell::new(x2w2)));
    //     x1w1x2w2.label = "x1w1x2w2".to_string();
    //
    //     let mut n = Value::add(Rc::new(RefCell::new(x1w1x2w2)), b.clone());
    //     n.label = "n".to_string();
    //
    //     /*let mut o = n.tanh();
    //     o.label = "o".to_string();
    //     println!("{}", o);
    //     o.grad = 1.0;
    //     o.backward();*/
    //     let e = Rc::new(RefCell::new(Value::exp(Rc::new(RefCell::new(Value::mul(Rc::new(RefCell::new(n.clone())), Rc::new(RefCell::new(Value::from(2)))))))));
    //     let o = Value::div(Rc::new(RefCell::new(Value::sub(e, Rc::new(RefCell::new(Value::from(1.0)))))), Rc::new(RefCell::new(Value::from(1.0))));
    //     //Value::exp(n.clone()).backward();
    //     //println!("{}", o);
    // }
    //
    #[test]
    fn test_backward() {
        let a = Value(Rc::new(RefCell::new(Value_ { data: 3.0, op: None, label: "a".to_string(), grad: 0.0 })));
        let mut b = a.clone().add(a.clone());
        b.as_deref_mut().grad = 1.0;
        b.backward();
        assert_eq!(a.clone().as_deref().grad, 2.0);


        //println!("{}", b);
    }
    //
    // #[test]
    // fn test_exp() {
    //     let a = Rc::new(RefCell::new(Value { data: 2.0, prev: vec![], op: None, label: "a".to_string(), grad: 0.0 }));
    //     let b = Value::exp(a);
    // }
}
