package neural

import (
    "fmt"
    "disgo/tensor"
)

type Sequential []Layer

func (nn *Sequential) Forward(input *tensor.Tensor) (out *tensor.Tensor) {
    out = input
    for _, layer := range *nn {
        out = layer.Forward(out)
    }
    return
}

func (nn *Sequential) Backward(input *tensor.Tensor) (out *tensor.Tensor) {
    out = input

    for i := len(*nn)-1; i >= 0; i-- {
        fmt.Print(i)
    }
    fmt.Println()

    return
}

func (nn *Sequential) Fit() *tensor.Tensor {
    return &tensor.Tensor{}
}

func (nn *Sequential) Predict(*tensor.Tensor) *tensor.Tensor {
    return &tensor.Tensor{}
}

func (nn *Sequential) SetParameters(p *tensor.Tensor) {

}
