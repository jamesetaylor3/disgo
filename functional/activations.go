package functional

import (
    "math"
    "disgo"
    "disgo/tensor"
)

type ActivationFunction struct {
	forward  func(*tensor.Tensor) *tensor.Tensor
	backward func(*tensor.Tensor) *tensor.Tensor
}

func (nn ActivationFunction) Forward(in *tensor.Tensor) *tensor.Tensor {
    return nn.forward(in)
}

func (nn ActivationFunction) Backward(in *tensor.Tensor) *tensor.Tensor {
    return nn.backward(in)
}

func IsNilActivation(a ActivationFunction) bool {
    if a.forward == nil { return true }
    if a.backward == nil { return true }
    return false
}

var Sigmoid ActivationFunction = ActivationFunction{
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        out = t.Copy()
        out.PiecewiseFunction(disgo.Sigmoid)
        return
    },
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        out = t.Copy()
        out.PiecewiseFunction(
            func (a float64) float64 {
                return disgo.Sigmoid(a) * (1 - disgo.Sigmoid(a))
            },
        )
        return
    },
}

var ReLU ActivationFunction = ActivationFunction{
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        out = t.Copy()
        out.PiecewiseFunction(disgo.ReLU)
        return
    },
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        out = t.Copy()
        out.PiecewiseFunction(
            func (a float64) float64 {
                if a >= 0 {
                    return 1
                } else {
                    return 0
                }
            },
        )
        return
    },
}

var ReLU6 ActivationFunction = ActivationFunction{
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        out = t.Copy()
        out.PiecewiseFunction(disgo.ReLU6)
        return
    },
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        out = t.Copy()
        out.PiecewiseFunction(
            func (a float64) float64 {
                if a >= 6 {
                    return 0
                } else if a >= 0 {
                    return 1
                } else {
                    return 0
                }
            },
        )
        return
    },
}

var LeakyReLU ActivationFunction = ActivationFunction{
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        out = t.Copy()
        out.PiecewiseFunction(disgo.LeakyReLU)
        return
    },
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        out = t.Copy()
        out.PiecewiseFunction(
            func (a float64) float64 {
                if a >= 0 {
                    return 1
                } else {
                    return 0.01
                }
            },
        )
        return
    },
}

var Tanh ActivationFunction = ActivationFunction{
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        out = tensor.Tanh(t)
        return
    },
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        out = t.Copy()
        out.PiecewiseFunction(
            func (a float64) float64 {
                return 1 - math.Pow(math.Tanh(a), 2)
            },
        )
        return
    },
}

var Softmax ActivationFunction = ActivationFunction{
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        out = t.Copy()
        tensor.AssertVector(t)
        sum := tensor.ScalarToFloat(tensor.Sum(tensor.Exp(t)))
        out.PiecewiseFunction(
            func (a float64) float64 {
                return math.Exp(a) / sum
            },
        )
        return
    },
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        tensor.AssertVector(t)
        out = t.Copy()
        out.PiecewiseFunction(
            func (a float64) float64 {
                return a * (1 - a)
            },
        )
        return
    },
}

var EmptyActivationFunction ActivationFunction = ActivationFunction{
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        return t
    },
    func(t *tensor.Tensor) (out *tensor.Tensor) {
        out = t.Copy()
        out.PiecewiseFunction(
            func (a float64) float64 {
                return 1
            },
        )
        return
    },
}
