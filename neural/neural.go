package neural

import (
    "disgo/tensor"
    F "disgo/functional"
)

type Layer interface {
    Forward(*tensor.Tensor) *tensor.Tensor
    Backward(*tensor.Tensor) *tensor.Tensor
    SetParameters(*tensor.Tensor)
}

func NewDeepNet(inSize, outSize, numHidden, hiddenSize int, activation F.ActivationFunction, bias bool) *Sequential {
    seq := make(Sequential, 1 + numHidden)

    seq[0] = &Linear{InSize: inSize, OutSize: hiddenSize, Bias: bias, Activation: activation}
    for i := 0; i < numHidden - 1; i++ {
        seq[i + 1] = &Linear{InSize: hiddenSize, OutSize: hiddenSize, Bias: bias, Activation: activation}
    }
    seq[numHidden] = &Linear{InSize: hiddenSize, OutSize: outSize, Bias: bias, Activation: activation}

    return &seq
}
