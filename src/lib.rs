use std::cell::{RefCell};
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
    // Add { lhs: Value, rhs: Value },
    // Sub { lhs: Value, rhs: Value },
    // Mul { lhs: Value, rhs: Value },
    // Div { lhs: Value, rhs: Value },
    // Pow { lhs: Value, rhs: Value },
    Unary(Value, UnaryOp),
    //    Tanh { x: Value },
    //    Exp { x: Value },
}

impl Op {
    pub fn backward(&self, out: &Value) -> (f64, f64) {
        match self {
            Binary(_, _, BinaryOp::Add) => (1.0 * out.as_deref().grad, 1.0 * out.as_deref().grad),
            Binary(lhs, rhs, BinaryOp::Mul) => (lhs.as_deref().data * out.as_deref().grad, rhs.as_deref().data * out.as_deref().grad),
            Binary(_, _, BinaryOp::Div) => unimplemented!(),
            Binary(_, _, BinaryOp::Sub) => unimplemented!(),
            Unary(_, UnaryOp::Tanh) => ((1.0 - out.as_deref().data.powi(2)) * out.as_deref().grad, 0.0),
            Unary(x, UnaryOp::Exp) => (x.as_deref().data.exp() * out.as_deref().grad, 0.0),
            Binary(lhs, rhs, BinaryOp::Pow) => (lhs.as_deref().data.powf(rhs.as_deref().data - 1.0) * rhs.as_deref().data * out.as_deref().grad, 0.0),
        }
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, Op::Binary { .. } )
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
            op: Some(Op::Unary(self.clone(), UnaryOp::Tanh)),
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
        let mut topo = Vec::new();
        let mut visted = HashMap::new();
        build_topo(self, &mut visted, &mut topo);

        self.as_deref_mut().grad = 1.0;
        for v in topo.iter().rev() {
            if let Some(op) = &v.as_deref().op {
                op.backward(self);
            }
        }
    }

    pub fn add(&self, other: Value) -> Value {
        Value(Rc::new(RefCell::new(Value_ {
            data: self.as_deref().data + other.as_deref().data,
            op: Some(Op::Binary(self.clone(), other.clone(), BinaryOp::Add)),
            label: "".to_string(),
            grad: 0.0,
        })))
    }

    pub fn mul(&self, other: Value) -> Value {
        Value(Rc::new(RefCell::new(Value_ {
            data: self.as_deref().data * other.as_deref().data,
            op: Some(Op::Binary(self.clone(), other.clone(), BinaryOp::Mul)),
            label: "".to_string(),
            grad: 0.0,
        })))
    }

    pub fn exp(&self) -> Value {
        Value(Rc::new(RefCell::new(Value_ {
            data: self.as_deref().data.exp(),
            op: Some(Op::Unary(self.clone(), UnaryOp::Exp)),
            label: "".to_string(),
            grad: 0.0,
        })))
    }

    pub fn pow(&self, other: Value) -> Value {
        Value(Rc::new(RefCell::new(Value_ {
            data: self.as_deref().data.powf(other.as_deref().data),
            op: Some(Op::Binary(self.clone(), other.clone(), BinaryOp::Pow)),
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
    #[test]
    fn test_tanh() {
        let x1 = Value(Rc::new(RefCell::new(Value_ { data: 2.0, op: None, label: "x1".to_string(), grad: 0.0 })));
        let x2 = Value(Rc::new(RefCell::new(Value_ { data: 0.0, op: None, label: "x2".to_string(), grad: 0.0 })));

        let w1 = Value(Rc::new(RefCell::new(Value_ { data: -3.0, op: None, label: "w1".to_string(), grad: 0.0 })));
        let w2 = Value(Rc::new(RefCell::new(Value_ { data: 1.0, op: None, label: "w2".to_string(), grad: 0.0 })));

        let b = Value(Rc::new(RefCell::new(Value_ { data: 6.8813735870195432, op: None, label: "b".to_string(), grad: 0.0 })));

        let x1w1 = x1.mul(w1.clone());
        x1w1.as_deref_mut().label = "x1w1".to_string();
        let x2w2 = x2.mul(w2.clone());
        x2w2.as_deref_mut().label = "x2w2".to_string();

        let x1w1x2w2 = x1w1.add(x2w2);
        x1w1x2w2.as_deref_mut().label = "x1w1x2w2".to_string();

        let n = x1w1x2w2.add(b.clone());
        n.as_deref_mut().label = "n".to_string();

        let mut o1 = n.tanh();
        o1.as_deref_mut().label = "o".to_string();
        println!("{:?}", o1);
        o1.as_deref_mut().grad = 1.0;
        o1.backward();

        let e = Value::from(2.0).mul(n).exp();
        let mut o = e.sub(Value::from(1.0)).div(e.add(Value::from(1.0)));
        o.as_deref_mut().label = "o".to_string();
        o.backward();
        println!("{:?}", o);
    }

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
