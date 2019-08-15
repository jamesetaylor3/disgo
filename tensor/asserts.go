package tensor

import (
    "fmt"
)

// Honestly these arent as helpful as i thought they would be

func Assert(b bool) {
    if !b {
        err := fmt.Errorf("Boolean assert error!")
        panic(err)
    }
}

func AssertLoc(s []int, l ...int) {
    for i, s_i := range s {
        if !(l[i] >= 0 && l[i] < s_i) {
            err := fmt.Errorf("Bad Loc! For Loc[%v], expecting value in between 0 and %v. Intead got %v!", i, s_i, l[i])
            panic(err)
        }
    }
}

func AssertDimension(t *Tensor, n int) {
    if t.dimension != n {
        err := fmt.Errorf("Expecting tensor dimension of %v but got dimension of %v!", n, t.dimension)
        panic(err)
    }
}

func AssertShape(t *Tensor, shape []int) {
    tmp := &Tensor{shape: shape}
    if !ShapeEq(t, tmp) {
        err := fmt.Errorf("Expecting tensor shape of %v but got shape of %v!", shape, t.shape)
        panic(err)
    }
}

func AssertSameShape(t, v *Tensor) {
    if !ShapeEq(t, v) {
        err := fmt.Errorf("Mismatching shapes! %v and %v are not the same!", t.shape, v.shape)
        panic(err)
    }
}

func AssertScalar(t *Tensor) {
    tmp := &Tensor{shape: []int{1}}
    if !ShapeEq(t, tmp) {
        err := fmt.Errorf("Expecting tensor shape of %v but got shape of %v!", tmp.shape, t.shape)
        panic(err)
    }
}

func AssertVector(t *Tensor) {
    if t.dimension != 1 {
        err := fmt.Errorf("Expecting vector dimension of 1 but got dimension of %v!", t.dimension)
        panic(err)
    }
}

func AssertMatrix(t *Tensor) {
    if t.dimension != 2 {
        err := fmt.Errorf("Expecting matrix dimension of 2 but got dimension of %v!", t.dimension)
        panic(err)
    }
}
