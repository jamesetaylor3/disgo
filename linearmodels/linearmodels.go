package linearmodels

import (
    "disgo"
)

type LinearModel interface {
    Optimize() disgo.Parameters
    ExportModel() func([]float64) float64
    Predict([][]float64) []float64
}

func AddBiasTermToInput(input []float64) []float64 {
    return append(input, 1)
}

func AddBiasTermToInputTable(input [][]float64) [][]float64 {
    for i := range input {
        input[i] = append(input[i], 1)
    }
    return input
}
