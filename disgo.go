package disgo

import (
    "math"
)

// things here don't have a permanent home

type Parameters []float64
type Gradient []float64

// Make this general model for linear models and neural nets
type Model interface{
    Forward([]float64) float64
    Backward([][]float64, []float64) Gradient
    Optimize() Parameters
    Predict([][]float64) []float64
    SetParameters(Parameters)
}

// Remove these once tensor library fully "functional" lol
func Sigmoid(x float64) float64 {
    return 1 / (1 + math.Exp(-x))
}

func ReLU(x float64) float64 {
    if x > 0 {
        return x
    } else {
        return 0
    }
}

func ReLU6(x float64) float64 {
    if x > 6 {
        return 6
    } else if x > 0 {
        return x
    } else {
        return 0
    }
}

func LeakyReLU(x float64) float64 {
    if x > 0 {
        return x
    } else {
        return 0.01 * x
    }
}

func Sgn(x float64) float64 {
    if x < 0 {
        return -1
    } else if x == 0 {
        return 0
    } else {
        return 1
    }
}
