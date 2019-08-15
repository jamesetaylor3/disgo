package neural

import (
    "disgo/tensor"
)

type ConvolutionalLayer struct {
    InChannels  int
    OutChannels int
    OutSize     int
    Stride      int
    Padding     int
    Dilation    int
    Bias        bool
    Dimension   int
    weights     *tensor.Tensor
    biases      *tensor.Tensor
}

func (l *ConvolutionalLayer) ForwardPass(prev *tensor.Tensor) (pass *tensor.Tensor) {
    if l.weights == nil {
        // l.weights = tensor.Rand(l.OutSize, l.InSize)
        // if l.Bias { l.biases = tensor.Rand(l.OutSize) }
    }

    pass = &tensor.Tensor{}
    return
}

func (l *ConvolutionalLayer) BackwardsPass(prev *tensor.Tensor) (pass *tensor.Tensor) {
    pass = &tensor.Tensor{}
    return
}
