package metrics

import (
    "math"
    "disgo"
)

type mse func([]float64, []float64) float64
type mae func([]float64, []float64) float64
type hl  func([]float64, []float64) float64
type lcl func([]float64, []float64) float64


var MeanSquaredError mse = func(target, pred []float64) float64 {
    var loss float64

    for i := 0; i < len(pred); i++ {
        loss += math.Pow(target[i] - pred[i], 2) / (2 * float64(len(pred)))
    }

    return loss
}

func (mse) Gradient(input [][]float64, pred, target []float64) disgo.Gradient {
    grad := make(disgo.Gradient, len(input[0]))

    for i := 0; i < len(input); i++ {
        for j := 0; j < len(input[i]); j++ {
            grad[j] += (pred[i] - target[i]) * input[i][j] / float64(len(input[i]))
        }
    }

    return grad
}

var MeanAbsoluteError mae = func(target, pred []float64) float64 {
    var loss float64

    for i := 0; i < len(target); i++ {
        loss += math.Abs(target[i] - pred[i]) / float64(len(pred))
    }

    return loss
}

func (mae) Gradient(input [][]float64, pred, target []float64, bias bool) disgo.Gradient {
    grad := make(disgo.Gradient, len(input[0]))

    for i := 0; i < len(input); i++ {
        for j := 0; j < len(input[i]); j++ {
            if pred[i] < target[i] {
                grad[j] -= 1 / float64(len(input[i]))
            } else if pred[i] > target[i] {
                grad[j] += 1 / float64(len(input[i]))
            }
        }
    }
    return grad
}


// Ones below are not finished

const DELTA = 0.5

var HuberLoss hl = func(target, pred []float64) float64 {
    var loss float64

    for i := 0; i < len(target); i++ {

        if math.Abs(target[i] - pred[i]) <= DELTA {
            loss += 0.5 * math.Pow(target[i] - pred[i], 2)
        } else {
            loss += DELTA * math.Abs(target[i] - pred[i]) - 0.5 * math.Pow(DELTA, 2)
        }

    }

    return loss
}

func (hl) Gradient(input [][]float64, pred, target []float64) disgo.Gradient {
    return disgo.Gradient{}
}


var LogCoshLoss lcl = func(target, pred []float64) float64 {
    var loss float64

    for i := 0; i < len(target); i++ {
        loss += math.Log(math.Cosh(pred[i] - target[i]))
    }

    return loss
}

func (lcl) Gradient(input [][]float64, pred, target []float64) disgo.Gradient {
    return disgo.Gradient{}
}

func EvaluateRegression(target, pred []float64) map[string]float64 {
    return map[string]float64{
        "MeanSquaredError": MeanSquaredError(target, pred),
        "MeanAbsoluteError": MeanAbsoluteError(target, pred),
    }
}
