package neural

import (
    "disgo/tensor"
    F "disgo/functional"
)

type Linear struct {
    InSize     int
    OutSize    int
    Bias       bool
    Activation F.ActivationFunction
    weights    *tensor.Tensor
    biases     *tensor.Tensor
    netsums    *tensor.Tensor
    LayerError float64  // This is only temporary until we figure out the better way to implement backpropogation
    LayerDelta float64  // same thing with this one here!
}

func (l *Linear) Forward(input *tensor.Tensor) (pass *tensor.Tensor) {

    if l.weights == nil {
        l.weights = tensor.Rand(l.OutSize, l.InSize)
        if l.Bias {
            l.biases = tensor.Rand(l.OutSize)
        }
        if F.IsNilActivation(l.Activation) {
            l.Activation = F.EmptyActivationFunction
        }
    }

    pass = tensor.MVProd(l.weights, input)

    l.netsums = pass.Copy()

    if l.Bias { pass = tensor.Add(pass, l.biases) }

    pass = l.Activation.Forward(pass)

    return
}

// http://code-spot.co.za/2009/10/08/15-steps-to-implemented-a-neural-net/

func (l *Linear) Backward(prev *tensor.Tensor) (pass *tensor.Tensor) {
    pass = prev

    dodn := l.Activation.Backward(l.netsums)

    dEdn := tensor.Hadamard(pass, dodn)

    // Maybe Hadamard with each row in the transposed weights matrix
    dEdw := tensor.Kronecker(l.weights.Transpose(), dEdn)

    pass = dEdw
    return
}

func (l *Linear) SetParameters(p *tensor.Tensor) {

}
