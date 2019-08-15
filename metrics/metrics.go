package metrics

import (
    "disgo"
)

type Criterion interface {
    Gradient([][]float64, []float64, []float64) disgo.Gradient
}

const EPSILON = 1e-8
