mod nn;

use std::collections::{HashMap};
use std::fmt::{Debug, Display};
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
    Relu,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Binary(Value, Value, BinaryOp),
    Unary(Value, UnaryOp),
}

impl Op {
    pub fn is_binary(&self) -> bool {
        matches!(self, Op::Binary { .. } )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueId(usize);

impl ValueId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Value_ {
    pub data: f64,
    pub op: Option<Op>,
    pub label: String,
    pub id: ValueId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GradStore(HashMap<ValueId, f64>);

impl Default for GradStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GradStore {
    pub fn new() -> GradStore {
        GradStore(HashMap::new())
    }

    pub fn or_insert(&mut self, id: ValueId) -> &mut f64 {
        self.0.entry(id).or_insert(0.0)
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
            id: ValueId::new(),
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

impl Default for Value {
    fn default() -> Self {
        Value(Rc::new(Value_ {
            data: 0.0,
            op: None,
            label: "".to_string(),
            id: ValueId::new(),
        }))
    }
}

macro_rules! bin_trait {
    ($trait:ident, $fn1:ident) => {
        impl<B: std::borrow::Borrow<Value>> std::ops::$trait<B> for Value {
            type Output = Value;

            fn $fn1(self, rhs: B) -> Self::Output {
                Value::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Value>> std::ops::$trait<B> for &Value {
            type Output = Value;

            fn $fn1(self, rhs: B) -> Self::Output {
                Value::$fn1(&self, rhs.borrow())
            }
        }

        impl std::ops::$trait<f64> for Value {
            type Output = Value;

            fn $fn1(self, rhs: f64) -> Self::Output {
                Value::$fn1(&self, &Value::from(rhs))
            }
        }

        impl std::ops::$trait<f64> for &Value {
            type Output = Value;

            fn $fn1(self, rhs: f64) -> Self::Output {
                Value::$fn1(&self, &Value::from(rhs))
            }
        }
    };
}

macro_rules! r_bin_trait {
    ($trait:ident, $fn1:ident, $op: tt) => {
        impl std::ops::$trait<Value> for f64 {
            type Output = Value;

            fn $fn1(self, rhs: Value) -> Self::Output {
                rhs $op self
            }
        }

        impl std::ops::$trait<&Value> for f64 {
            type Output = Value;

            fn $fn1(self, rhs: &Value) -> Self::Output {
                rhs $op self
            }
        }
    };
}

bin_trait!(Add, add);
bin_trait!(Mul, mul);
bin_trait!(Sub, sub);
bin_trait!(Div, div);

r_bin_trait!(Add, add, +);
r_bin_trait!(Mul, mul, *);
r_bin_trait!(Sub, sub, -);
r_bin_trait!(Div, div, /);


impl std::ops::Neg for Value {
    type Output = Value;

    fn neg(self) -> Value {
        Value::neg(&self)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value[{}]", self.data)
    }
}

impl Value {
    pub fn new_with_label(data: f64, label: &str) -> Value {
        Value(Rc::new(Value_ {
            data,
            op: None,
            label: label.to_string(),
            id: ValueId::new(),
        }))
    }

    pub fn new(data: f64) -> Value {
        Value(Rc::new(Value_ {
            data,
            op: None,
            label: "".to_string(),
            id: ValueId::new(),
        }))
    }
    pub fn tanh(&self) -> Value {
        let x = self.data;
        let t = ((x * 2.0).exp() - 1.0) / ((x * 2.0).exp() + 1.0);
        Value(Rc::new(Value_ {
            data: t,
            op: Some(Op::Unary(self.clone(), UnaryOp::Tanh)),
            label: "".to_string(),
            id: ValueId::new(),
        }))
    }

    pub fn backward(&self) -> GradStore {
        fn build_topo(value: &Value, visited: &mut HashMap<ValueId, bool>, topo: &mut Vec<Value>) {
            if visited.contains_key(&value.id) {
                return;
            }
            visited.insert(value.id, true);
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
        grad_store.0.insert(self.id, 1.0);
        let mut topo = Vec::new();
        let mut visted = HashMap::new();
        build_topo(self, &mut visted, &mut topo);

        for v in topo.iter().rev() {
            let v_grad = *grad_store.0.get(&v.id).unwrap();

            if let Some(op) = &v.op {
                match op {
                    Binary(lhs, rhs, BinaryOp::Add) => {
                        let g = grad_store.or_insert(lhs.id);
                        *g += v_grad;
                        let g = grad_store.or_insert(rhs.id);
                        *g += v_grad;
                    }
                    Binary(lhs, rhs, BinaryOp::Mul) => {
                        let g = grad_store.or_insert(lhs.id);
                        *g += rhs.data * v_grad;
                        let g = grad_store.or_insert(rhs.id);
                        *g += lhs.data * v_grad;
                    }
                    Binary(_, _, BinaryOp::Div) => {
                        unreachable!()
                    }
                    Binary(_, _, BinaryOp::Sub) => {
                        unreachable!()
                    }
                    Binary(lhs, rhs, BinaryOp::Pow) => {
                        let g = grad_store.or_insert(lhs.id);
                        *g += rhs.data * lhs.data.powf(rhs.data - 1.0) * v_grad;
                        let g = grad_store.or_insert(rhs.id);
                        *g += lhs.data.powf(rhs.data) * v_grad;
                    }
                    Unary(x, UnaryOp::Tanh) => {
                        let g = grad_store.or_insert(x.id);
                        *g += (1.0 - v.data.powi(2)) * v_grad;
                    }
                    Unary(x, UnaryOp::Exp) => {
                        let g = grad_store.or_insert(x.id);
                        *g += v.data.exp() * v_grad;
                    }
                    Unary(x, UnaryOp::Relu) => {
                        let g = grad_store.or_insert(x.id);
                        *g += (x.data > 0.0) as i32 as f64 * v_grad;
                    }
                }
            }
        }
        grad_store
    }

    pub fn add(&self, other: &Value) -> Value {
        Value(Rc::new(Value_ {
            data: self.data + other.data,
            op: Some(Op::Binary(self.clone(), other.clone(), BinaryOp::Add)),
            label: "".to_string(),
            id: ValueId::new(),
        }))
    }

    pub fn mul(&self, other: &Value) -> Value {
        Value(Rc::new(Value_ {
            data: self.data * other.data,
            op: Some(Op::Binary(self.clone(), other.clone(), BinaryOp::Mul)),
            label: "".to_string(),
            id: ValueId::new(),
        }))
    }

    pub fn exp(&self) -> Value {
        Value(Rc::new(Value_ {
            data: self.data.exp(),
            op: Some(Op::Unary(self.clone(), UnaryOp::Exp)),
            label: "".to_string(),
            id: ValueId::new(),
        }))
    }

    pub fn pow(&self, other: &Value) -> Value {
        Value(Rc::new(Value_ {
            data: self.data.powf(other.data),
            op: Some(Op::Binary(self.clone(), other.clone(), BinaryOp::Pow)),
            label: "".to_string(),
            id: ValueId::new(),
        }))
    }

    pub fn sqrt(&self) -> Value {
        self.mul(&self)
    }

    pub fn div(&self, other: &Value) -> Value {
        self.mul(&other.pow(&Value::from(-1.0)))
    }

    pub fn neg(&self) -> Value {
        self.mul(&Value::from(-1.0))
    }

    pub fn sub(&self, other: &Value) -> Value {
        self.add(&other.neg())
    }

    pub fn relu(&self) -> Value {
        Value(Rc::new(Value_ {
            data: self.data.max(0.0),
            op: None,
            label: "ReLU".to_string(),
            id: ValueId::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tanh() {
        let x1 = Value::new_with_label(2.0, "x1");
        let x2 = Value::new_with_label(0.0, "x2");

        let w1 = Value::new_with_label(-3.0, "w1");
        let w2 = Value::new_with_label(1.0, "w2");


        let b = Value::new_with_label(6.8813735870195432, "b");

        let mut x1w1 = x1 * w1;
        x1w1.label = "x1w1".to_string();
        let mut x2w2 = x2 * w2;
        x2w2.label = "x2w2".to_string();

        let mut x1w1x2w2 = x1w1 + x2w2;
        x1w1x2w2.label = "x1w1x2w2".to_string();

        let mut n = x1w1x2w2 + b;
        n.label = "n".to_string();

        let mut o1 = n.tanh();
        o1.label = "o".to_string();
        println!("{:?}", o1);
        o1.backward();

        let e = (n * 2.0_f64).exp();
        let mut o = (&e - 1.0_f64) / (e + 1.0_f64);
        o.label = "o".to_string();
        o.backward();
        println!("{:?}", o);
    }

    #[test]
    fn test_backward() {
        let a = Value::new_with_label(3.0, "a");
        let b = a.clone() + a.clone();
        let g = b.backward();
        assert_eq!(*(g.0.get(&a.id).unwrap()), 2.0);
    }
}
