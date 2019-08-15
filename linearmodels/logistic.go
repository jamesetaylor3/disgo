package linearmodels

import (
    "disgo"
    "disgo/optim"
    "disgo/metrics"
)

const (
    DEFAULT_ACTIVATION_THRESHOLD = 0.5
)

// add multiorder at some point
type LogisticRegressionModel struct {
    Input [][]float64
    Target []float64
    Criterion metrics.Criterion
    Optimizer optim.Optimizer
    Parameters disgo.Parameters
    Bias bool
    dimension int
}

func (m *LogisticRegressionModel) Optimize() disgo.Parameters {

    if m.Bias { m.Input = AddBiasTermToInputTable(m.Input) }

    m.dimension = len(m.Input[0])

    m.Parameters = make([]float64, len(m.Input[0]))

    parameters := optim.SGD{
        Input: m.Input,
        Target: m.Target,
        Optimizer: m.Optimizer,
        Dimension: m.dimension,
        Loud: true,
        Model: m,
    }.Run()

    return parameters
}

func (m *LogisticRegressionModel) Predict(test_input [][]float64) []float64 {

    predictions := make([]float64, len(test_input))

    for i := 0; i < len(test_input); i++ {
        predictions[i] = m.Forward(test_input[i])
    }

    return predictions

}

func (m *LogisticRegressionModel) Forward(xs []float64) float64 {
    var y float64

    if len(xs) != len(m.Input[0]) { xs = AddBiasTermToInput(xs) }

    for i, x_i := range xs {
        y += x_i * m.Parameters[i]
    }

    return disgo.Sigmoid(y)
}

func (m *LogisticRegressionModel) Backward(input [][]float64, target []float64) disgo.Gradient {
    pred := make([]float64, len(target))

    for i := range input {
        pred[i] = m.Forward(input[i])
    }

    return m.Criterion.Gradient(input, pred, target)
}

func (m *LogisticRegressionModel) SetParameters(p disgo.Parameters) {
    m.Parameters = p
}
