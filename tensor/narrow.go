package tensor

import (
    "fmt"
)

func (t *Tensor) MatrixNarrow1stD(a, b int) {

}

func MatrixShuffle1stDWithTarget(train, target *Tensor) {

}

// combine narrow and slice in some way
// Need this though to implement into SGD and stuff
// MAKE IT ONLY FOR VECTORS and MATRICES so that we can get the show on the road with implementing tenors in SGD

// Slice{2, 4} // slice in located dimension from 2 to 4
// Slice{-1} means all in that dimension
// Exact = 4 only fourth element in that dimesion
type Slice [2]int
type Exact [1]int

type Narrower interface{
    Val() interface{}
}

func (s Slice) Val() interface{} {
    return s
}

func (e Exact) Val() interface{} {
    return e
}

func sliceEqual(a, b []int) bool {
    if len(a) != len(b) {
        return false
    }
    for i, v := range a {
        if v != b[i] {
            return false
        }
    }
    return true
}

func (t *Tensor) Narrow2(ns ...Narrower) *Tensor {

    for _, n := range ns {
        switch n.(type) {
        case Slice:
            fmt.Printf("%T, %v\n", n.Val().(Slice), n.Val().(Slice))
        case Exact:
            fmt.Printf("%T, %v\n", n.Val().(Exact), n.Val().(Exact))
        }
    }

    return &Tensor{}
}

// This might be the shittiest piece of code i have written in my life what the fuck
func (t *Tensor) Narrow(l ...int) *Tensor {

    var (
        slicedData         []float64
        slicedShape        []int
        indexPossibilities [][]int
        firstPossibilities []int
    )

    for _, l_i := range l {
        if l_i == -1 {
            firstPossibilities = append(firstPossibilities, 0)
        } else {
            firstPossibilities = append(firstPossibilities, l_i)
        }
    }
    indexPossibilities = append(indexPossibilities, firstPossibilities)

    firstLine := true
    for i, l_i := range l {
        if l_i == -1 {
            slicedShape = append(slicedShape, t.shape[i])
            for j := 0; j < t.shape[i]; j++ {
                newIndexPossibilities := make([][]int, len(indexPossibilities))
                for k := range newIndexPossibilities {
                    tmp := make([]int, len(l))
                    copy(tmp, indexPossibilities[k])
                    newIndexPossibilities[k] = tmp
                    if firstLine {
                        firstLine = false
                        break
                    }
                    newIndexPossibilities[k][i] = j
                    if sliceEqual(newIndexPossibilities[k], indexPossibilities[len(indexPossibilities)-1]) { break }
                    indexPossibilities = append(indexPossibilities, newIndexPossibilities[k])
                }
            }
        }
    }
    for i := range indexPossibilities {
        slicedData = append(slicedData, *t.Loc(indexPossibilities[i]...))
    }

    return NewTensor(slicedData, slicedShape...)
}
