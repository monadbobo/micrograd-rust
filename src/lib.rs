use std::collections::{HashMap};
use std::fmt::{Debug};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use crate::Op::{Binary, Unary};

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Tanh,
    Exp,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Binary(Value, Value, BinaryOp),
    Unary(Value, UnaryOp),
}

impl Op {
    // pub fn backward(&self, out: &Value) -> (f64, f64) {
    //     match self {
    //         Binary(_, _, BinaryOp::Add) => (1.0 * out.grad, 1.0 * out.grad),
    //         Binary(lhs, rhs, BinaryOp::Mul) => (lhs.data * out.grad, rhs.data * out.grad),
    //         Binary(_, _, BinaryOp::Div) => unimplemented!(),
    //         Binary(_, _, BinaryOp::Sub) => unimplemented!(),
    //         Unary(_, UnaryOp::Tanh) => ((1.0 - out.data.powi(2)) * out.grad, 0.0),
    //         Unary(x, UnaryOp::Exp) => (x.data.exp() * out.grad, 0.0),
    //         Binary(lhs, rhs, BinaryOp::Pow) => (lhs.data.powf(rhs.data - 1.0) * rhs.data * out.grad, 0.0),
    //     }
    // }

    pub fn is_binary(&self) -> bool {
        matches!(self, Op::Binary { .. } )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Value_ {
    pub data: f64,
    pub op: Option<Op>,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GradStore(HashMap<String, f64>);

impl Default for GradStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GradStore {
    pub fn new() -> GradStore {
        GradStore(HashMap::new())
    }

    pub fn or_insert(&mut self, label: String) -> &mut f64 {
        self.0.entry(label).or_insert(0.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Value(Rc<Value_>);

impl Eq for Value {}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value(Rc::new(Value_ {
            data: value,
            op: None,
            label: "".to_string(),
        }))
    }
}

impl AsRef<Value> for Value {
    fn as_ref(&self) -> &Value {
        self
    }
}

impl Deref for Value {
    type Target = Value_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl DerefMut for Value {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Rc::make_mut(&mut self.0)
    }
}

impl Value {
    pub fn tanh(&self) -> Value {
        let x = self.data;
        let t = ((x * 2.0).exp() - 1.0) / ((x * 2.0).exp() + 1.0);
        Value(Rc::new(Value_ {
            data: t,
            op: Some(Op::Unary(self.clone(), UnaryOp::Tanh)),
            label: "".to_string(),
        }))
    }

    pub fn backward(&self) -> GradStore {
        fn build_topo(value: &Value, visited: &mut HashMap<String, bool>, topo: &mut Vec<Value>) {
            if visited.contains_key(&value.label) {
                return;
            }
            visited.insert(value.label.clone(), true);
            if let Some(op) = &value.op {
                match op {
                    Op::Binary(lhs, rhs, _) => {
                        build_topo(lhs, visited, topo);
                        build_topo(rhs, visited, topo);
                    }
                    Op::Unary(x, _) => {
                        build_topo(x, visited, topo);
                    }
                }
            }
            topo.push(value.clone());
        }
        let mut grad_store = GradStore::new();
        grad_store.0.insert(self.label.clone(), 1.0);
        let mut topo = Vec::new();
        let mut visted = HashMap::new();
        build_topo(self, &mut visted, &mut topo);

        for v in topo.iter().rev() {
            let v_grad = *grad_store.0.get(&v.label).unwrap();

            if let Some(op) = &v.op {
                match op {
                    Binary(lhs, rhs, BinaryOp::Add) => {
                        let g = grad_store.or_insert(lhs.label.clone());
                        *g += v_grad;
                        let g = grad_store.or_insert(rhs.label.clone());
                        *g += v_grad;
                    }
                    Binary(lhs, rhs, BinaryOp::Mul) => {
                        let g = grad_store.or_insert(lhs.label.clone());
                        *g += rhs.data * v_grad;
                        let g = grad_store.or_insert(rhs.label.clone());
                        *g += lhs.data * v_grad;
                    }
                    Binary(_, _, BinaryOp::Div) => {
                        unreachable!()
                    }
                    Binary(_, _, BinaryOp::Sub) => {
                        unreachable!()
                    }
                    Binary(lhs, rhs, BinaryOp::Pow) => {
                        let g = grad_store.or_insert(lhs.label.clone());
                        *g += rhs.data * lhs.data.powf(rhs.data - 1.0) * v_grad;
                        let g = grad_store.or_insert(rhs.label.clone());
                        *g += lhs.data.powf(rhs.data) * v_grad;
                    }
                    Unary(x, UnaryOp::Tanh) => {
                        let g = grad_store.or_insert(x.label.clone());
                        *g += (1.0 - v.data.powi(2)) * v_grad;
                    }
                    Unary(x, UnaryOp::Exp) => {
                        let g = grad_store.or_insert(x.label.clone());
                        *g += v.data.exp() * v_grad;
                    }
                }
            }
        }
        grad_store
    }

    pub fn add(&self, other: Value) -> Value {
        Value(Rc::new(Value_ {
            data: self.data + other.data,
            op: Some(Op::Binary(self.clone(), other.clone(), BinaryOp::Add)),
            label: "".to_string(),
        }))
    }

    pub fn mul(&self, other: Value) -> Value {
        Value(Rc::new(Value_ {
            data: self.data * other.data,
            op: Some(Op::Binary(self.clone(), other.clone(), BinaryOp::Mul)),
            label: "".to_string(),
        }))
    }

    pub fn exp(&self) -> Value {
        Value(Rc::new(Value_ {
            data: self.data.exp(),
            op: Some(Op::Unary(self.clone(), UnaryOp::Exp)),
            label: "".to_string(),
        }))
    }

    pub fn pow(&self, other: Value) -> Value {
        Value(Rc::new(Value_ {
            data: self.data.powf(other.data),
            op: Some(Op::Binary(self.clone(), other.clone(), BinaryOp::Pow)),
            label: "".to_string(),
        }))
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
    #[test]
    fn test_tanh() {
        let x1 = Value(Rc::new(Value_ { data: 2.0, op: None, label: "x1".to_string() }));
        let x2 = Value(Rc::new(Value_ { data: 0.0, op: None, label: "x2".to_string() }));

        let w1 = Value(Rc::new(Value_ { data: -3.0, op: None, label: "w1".to_string() }));
        let w2 = Value(Rc::new(Value_ { data: 1.0, op: None, label: "w2".to_string() }));

        let b = Value(Rc::new(Value_ { data: 6.8813735870195432, op: None, label: "b".to_string() }));

        let mut x1w1 = x1.mul(w1.clone());
        x1w1.label = "x1w1".to_string();
        let mut x2w2 = x2.mul(w2.clone());
        x2w2.label = "x2w2".to_string();

        let mut x1w1x2w2 = x1w1.add(x2w2);
        x1w1x2w2.label = "x1w1x2w2".to_string();

        let mut n = x1w1x2w2.add(b.clone());
        n.label = "n".to_string();

        let mut o1 = n.tanh();
        o1.label = "o".to_string();
        println!("{:?}", o1);
        o1.backward();

        let e = Value::from(2.0).mul(n).exp();
        let mut o = e.sub(Value::from(1.0)).div(e.add(Value::from(1.0)));
        o.label = "o".to_string();
        o.backward();
        println!("{:?}", o);
    }

    #[test]
    fn test_backward() {
        let a = Value(Rc::new(Value_ { data: 3.0, op: None, label: "a".to_string() }));
        let b = a.clone().add(a.clone());
        let g = b.backward();
        assert_eq!(*(g.0.get("a").unwrap()), 2.0);
    }
}
