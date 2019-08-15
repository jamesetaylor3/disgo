package metrics

import (
    "math"
    "disgo"
)


type cel func([]float64, []float64) float64

var CrossEntropyLoss cel = func(target, pred []float64) float64 {
    var loss float64

    for i := 0; i < len(pred); i++ {

        switch target[i] {
        case 1:
            loss += -1 * math.Log(pred[i] + EPSILON) / float64(len(pred))
        case 0:
            loss += -1 * math.Log(1 - pred[i] + EPSILON) / float64(len(pred))
        }

    }

    return loss
}

func (cel) Gradient(input [][]float64, pred, target []float64) disgo.Gradient {
    grad := make(disgo.Gradient, len(input[0]))

    for i := 0; i < len(input); i++ {
        for j := 0; j < len(input[i]); j++ {
            grad[j] += (pred[i] - target[i]) * input[i][j] / float64(len(input[i]))
        }
    }

    return grad
}

func ClassificationAccuracy(target, pred []float64) float64 {
    var (
        pred_i float64
        numAccurate int
    )

    for i := range target {
        if pred[i] > 0.5 {
            pred_i = 1
        } else {
            pred_i = 0
        }
        if pred_i == target[i] { numAccurate++ }
    }

    return float64(numAccurate) / float64(len(target))

}

// Have ability to create confusion matrix and get the other loss functions

func EvaluateClassification(target, pred []float64) map[string]float64 {
    return map[string]float64{
        "CrossEntropyLoss": CrossEntropyLoss(target, pred),
        "ClassificationAccuracy": ClassificationAccuracy(target, pred),
    }
}
