package tensor

import (
    "fmt"
    "time"
    "math/rand"
    // "strings"
)

// https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/
// This is all dense... make sparse
// Also optimize current slice function
// Be able to actually slice it and we are goldenk,

type Tensor struct {
    data         []float64
    shape        []int
    dimension    int
    dataaccessor []int
}

func (t *Tensor) Shape() []int {
    shape := make([]int, len(t.shape))
    copy(shape, t.shape)
    return shape
}

func (t *Tensor) Dimension() int {
    return t.dimension
}

func Zeros(s ...int) *Tensor {
    sizeData := 1
    currentDataAccessVal := 1
    dataaccessor := make([]int, len(s))

    for i, s_i := range s {
        sizeData *= s_i
        dataaccessor[i] = currentDataAccessVal
        currentDataAccessVal *= s_i
    }
    return &Tensor{
        data: make([]float64, sizeData),
        shape: s,
        dimension: len(s),
        dataaccessor: dataaccessor,
    }
}

func Rand(s ...int) *Tensor {
    t := Zeros(s...)

    rand.Seed(time.Now().UnixNano())

    for i := range t.data {
        t.data[i] = rand.NormFloat64()
    }

    return t
}

// have it input data
func NewTensor(data []float64, s ...int) *Tensor {
    t := Zeros(s...)
    t.data = data
    return t
}

func (t *Tensor) Loc(l ...int) *float64 {

    var sliceIndex int

    for i := range l {
        if !(l[i] >= 0 && l[i] < t.shape[i]) {
            err := fmt.Errorf("Bad Loc! For Loc[%v], expecting value in between 0 and %v. Intead got %v!", i, t.shape[i]-1, l[i])
            panic(err)
        }
        sliceIndex += l[i] * t.dataaccessor[i]
    }

    return &t.data[sliceIndex]
}

// Convert [4] -> [[4]]
func WrapOuter(t *Tensor) *Tensor {
    return &Tensor{}
}

func (t *Tensor) Copy() *Tensor {
    cpydata := make([]float64, len(t.data))
    copy(cpydata, t.data)

    return &Tensor{
        data: cpydata,
        shape: t.shape,
        dimension: t.dimension,
        dataaccessor: t.dataaccessor,
    }
}

func (t *Tensor) PiecewiseFunction(fn func (float64) float64) {
    for i := range t.data {
        t.data[i] = fn(t.data[i])
    }
}

func Dot(p, q *Tensor) (out *Tensor) {
    // AssertVector(p)
    // AssertVector(q)
    AssertSameShape(p, q)

    out = Sum(Hadamard(p, q))
    return
}

func Sum(t *Tensor) (out *Tensor) {
    out = Zeros(1)
    for _, val := range t.data {
        out.data[0] += val
    }
    return
}

func ScalarToFloat(t *Tensor) float64 {
    AssertScalar(t)
    return t.data[0]
}

// Need slicing for this and will need this for neural nets
func MVProd(m, v *Tensor) (out *Tensor) {
    AssertMatrix(m)
    AssertVector(v)

    out = Zeros(m.shape[0])

    for i := 0; i < m.shape[0]; i++ {
        *out.Loc(i) = ScalarToFloat(Dot(v, m.Narrow(i, -1)))
    }
    return
}

// Need to make sure it is either matrix or vector
func (t *Tensor) Transpose() (out *Tensor) {
    out = Zeros(t.shape[1], t.shape[0])
    for i := 0; i < t.shape[0]; i++ {
        for j := 0; j < t.shape[1]; j++ {
            *out.Loc(j, i) = *t.Loc(i, j)
        }
    }
    return
}

func (t *Tensor) Clamp(min, max float64) (out *Tensor) {
    out = t.Copy()
    for i := range out.data {
        if out.data[i] > max {
            out.data[i] = max
        } else if out.data[i] < min {
            out.data[i] = min
        }
    }
    return
}

func Hadamard(t, v *Tensor) (out *Tensor) {
    AssertSameShape(t, v)

    out = Zeros(t.shape...)
    for i := range out.data {
        out.data[i] = t.data[i] * v.data[i]
    }
    return
}

func Kronecker(t, v *Tensor) *Tensor {
    return &Tensor{}
}

// Only working with 2d tensors horizontally. must create way to make it more representative
func Concatenate(t, v *Tensor, dim int) *Tensor {
    out := Zeros(t.shape[0], t.shape[1] + v.shape[1])

    var feedLeft bool = true
    var leftTensorIndex, rightTensorIndex int

    for i := range out.data {
        if feedLeft {
            out.data[i] = t.data[leftTensorIndex]
            leftTensorIndex++
            if leftTensorIndex % t.shape[1] == 0 { feedLeft = false }
        } else {
            out.data[i] = v.data[rightTensorIndex]
            rightTensorIndex++
            if rightTensorIndex % v.shape[1] == 0 { feedLeft = true }
        }
    }

    return out
}

// Do string represenation woohooo
/**
func (tensor *Tensor) String() string {
    var string_representation strings.Builder

    string_representation.WriteString("[ [")

    for i := 0; i < tensor.shape[1]; i++ {
        for j := 0; j < tensor.shape[0]; j++ {
            if i > 0 && j == 0 {
                string_representation.WriteString("   ")
            }
            val_str := fmt.Sprintf("%f", *tensor.Loc(i, j))
            string_representation.WriteString(val_str)
            if j == (tensor.shape[1] - 1) { break }
            string_representation.WriteString(", ")
        }
        string_representation.WriteString("]")
        if i == (tensor.shape[0] - 1) { break }
        string_representation.WriteString("\n")

    }

    string_representation.WriteString(" ]")

    return string_representation.String()
}
**/
