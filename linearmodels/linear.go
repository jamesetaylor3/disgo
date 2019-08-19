package linearmodels

import (
    "disgo"
    "disgo/optim"
    "disgo/metrics"
)

type LinearRegressionModel struct {
    // Input      [][]float64
    // Target     []float64
    Criterion  metrics.Criterion
    Optimizer  optim.Optimizer
    Parameters disgo.Parameters
    Bias       bool
    inputDim   int
}

func (m *LinearRegressionModel) Fit(input [][]float64, target []float64, maxIterations int) disgo.Parameters {

    if m.Bias { input = AddBiasTermToInputTable(input) }

    m.inputDim = len(input[0])

    m.Parameters = make(disgo.Parameters, m.inputDim)

    parameters := optim.SGD{
        Dimension: m.inputDim,
        Input: input,
        Target: target,
        Loud: true,
        Optimizer: m.Optimizer,
        Model: m,
        MaxIterations: maxIterations,
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

    if len(xs) != m.inputDim { xs = AddBiasTermToInput(xs) }

    for i := range m.Parameters {
        y += xs[i] * m.Parameters[i]
    }

    return y
}

func (m *LinearRegressionModel) Backward(input [][]float64, target []float64) disgo.Gradient {
    pred := make([]float64, len(target))

    for i := range input {
        pred[i] = m.Forward(input[i])
    }

    if m.Bias {
        input = AddBiasTermToInputTable(input)
    }

    return m.Criterion.Gradient(input, pred, target)
}

func (m *LinearRegressionModel) SetParameters(p disgo.Parameters) {
    m.Parameters = p
}
