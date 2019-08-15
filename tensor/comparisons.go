package tensor

func ShapeEq(t, v *Tensor) bool {
    for i := range t.shape {
        if t.shape[i] != v.shape[i] { return false }
    }
    return true
}

func Equal(t, v *Tensor) bool {
    if len(t.data) != len(v.data) { return false }
    if !ShapeEq(t, v) { return false }
    for i := range t.data {
        if t.data[i] != v.data[i] { return false }
    }
    return true
}

func Ge(t, v *Tensor) *Tensor {
    data := make([]float64, len(t.data))
    var ge float64
    for i := range data {
        ge = 0
        if t.data[i] >= v.data[i] { ge = 1 }
        data[i] = ge
    }

    return &Tensor{
        data: data,
        shape: t.shape,
        dataaccessor: t.dataaccessor,
    }
}

func Gt(t, v *Tensor) *Tensor {
    data := make([]float64, len(t.data))
    var gt float64
    for i := range data {
        gt = 0
        if t.data[i] > v.data[i] { gt = 1 }
        data[i] = gt
    }

    return &Tensor{
        data: data,
        shape: t.shape,
        dataaccessor: t.dataaccessor,
    }
}

func Le(t, v *Tensor) *Tensor {
    data := make([]float64, len(t.data))
    var le float64
    for i := range data {
        le = 0
        if t.data[i] <= v.data[i] { le = 1 }
        data[i] = le
    }

    return &Tensor{
        data: data,
        shape: t.shape,
        dataaccessor: t.dataaccessor,
    }
}

func Lt(t, v *Tensor) *Tensor {
    data := make([]float64, len(t.data))
    var lt float64
    for i := range data {
        lt = 0
        if t.data[i] < v.data[i] { lt = 1 }
        data[i] = lt
    }

    return &Tensor{
        data: data,
        shape: t.shape,
        dataaccessor: t.dataaccessor,
    }
}
