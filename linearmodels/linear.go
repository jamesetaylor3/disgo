package linearmodels

import (
    "disgo"
    "disgo/optim"
    "disgo/metrics"
)

type LinearRegressionModel struct {
    Input [][]float64
    Target []float64
    Criterion metrics.Criterion
    Optimizer optim.Optimizer
    Parameters disgo.Parameters
    Bias bool
    dimension int
}

func (m *LinearRegressionModel) Optimize() disgo.Parameters {

    if m.Bias { m.dimension = len(m.Input[0]) + 1 } else { m.dimension = len(m.Input[0]) }

    m.Parameters = make(disgo.Parameters, m.dimension)

    parameters := optim.SGD{
        Dimension: m.dimension,
        Input: m.Input,
        Target: m.Target,
        Loud: true,
        Optimizer: m.Optimizer,
        Model: m,
    }.Run()

    return parameters
}

func (m *LinearRegressionModel) Predict(test_input [][]float64) []float64 {

    predictions := make([]float64, len(test_input))

    for i := 0; i < len(test_input); i++ {
        predictions[i] = m.Forward(test_input[i])
    }

    return predictions

}

func (m *LinearRegressionModel) Forward(xs []float64) float64 {
    var y float64

    if m.Bias { xs = AddBiasTermToInput(xs) }

    for i, x_i := range xs {
        y += x_i * m.Parameters[i]
    }

    // if m.Bias { y += m.Parameters[len(m.Parameters)-1] }

    return y
}

func (m *LinearRegressionModel) Backward(input [][]float64, target []float64) disgo.Gradient {
    pred := make([]float64, len(target))

    for i := range input {
        pred[i] = m.Forward(input[i])
    }

    input = AddBiasTermToInputTable(input)

    return m.Criterion.Gradient(input, pred, target)
}

func (m *LinearRegressionModel) SetParameters(p disgo.Parameters) {
    m.Parameters = p
}
