package minimize

import (
    "math"
    "disgo"
)

func Himmeleblau(p disgo.Parameters) float64 {
    return math.Pow(math.Pow(p[0], 2) + p[1] - 11, 2) + math.Pow(p[0] + math.Pow(p[1], 2) - 7, 2)
}

func Booth(p disgo.Parameters) float64 {
    return math.Pow(p[0] + 2 * p[1] - 7, 2) + math.Pow(2 * p[0] + p[1] - 5, 2)
}

func Mccormick(p disgo.Parameters) float64 {
    return math.Sin(p[0] + p[1]) + math.Pow(p[0] - p[1], 2) - 1.5 * p[0] + 2.5 * p[1] + 1
}

func F(p disgo.Parameters) float64 {
    return math.Pow(p[0] - 10, 2)
}

func G(p disgo.Parameters) disgo.Gradient {
    return disgo.Gradient{2 * (p[0] - 10)}
}

func Banana(p disgo.Parameters) float64 {
    return math.Pow(1 - p[0], 2) + 100 * math.Pow(p[1] - math.Pow(p[0], 2), 2)
}
