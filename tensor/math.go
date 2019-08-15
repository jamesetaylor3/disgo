package tensor

import (
    "math"
)

// Basically a port of math package in go but missing some

func Abs(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Abs)
    return
}

func Acos(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Acos)
    return
}

func Add(t, v *Tensor) (out *Tensor) {
    AssertSameShape(t, v)
    out = Zeros(t.shape...)
    for i := range out.data {
        out.data[i] = t.data[i] + v.data[i]
    }
    return
}

func Asin(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Asin)
    return
}

func Atan(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Atan)
    return
}

func Ceil(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Ceil)
    return
}

func Cos(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Cos)
    return
}

func Cosh(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Cosh)
    return
}

func Div(t *Tensor, v float64) (out *Tensor) {
    div := func(a float64) float64 {
        return a / v
    }
    out = t.Copy()
    out.PiecewiseFunction(div)
    return
}

func Exp(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Exp)
    return
}

func Log(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Log)
    return
}

func Mul(t *Tensor, v float64) (out *Tensor) {
    mul := func(a float64) float64 {
        return a * v
    }
    out = t.Copy()
    out.PiecewiseFunction(mul)
    return
}

func Neg(t *Tensor) (out *Tensor) {
    out = Mul(t, -1)
    return
}

func Pow(t *Tensor, v float64) (out *Tensor) {
    pow := func(a float64) float64 {
        return math.Pow(a, v)
    }
    out = t.Copy()
    out.PiecewiseFunction(pow)
    return
}

func Sin(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Sin)
    return
}

func Sinh(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Sinh)
    return
}

func Sqrt(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Sqrt)
    return
}

func Sub(t, v *Tensor) (out *Tensor) {
    AssertSameShape(t, v)
    out = Zeros(t.shape...)
    for i := range out.data {
        out.data[i] = t.data[i] - v.data[i]
    }
    return
}

func Tan(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Tan)
    return
}

func Tanh(t *Tensor) (out *Tensor) {
    out = t.Copy()
    out.PiecewiseFunction(math.Tanh)
    return
}
