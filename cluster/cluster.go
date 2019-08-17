package cluster

import (
    "math"
    "time"
    "math/rand"
)

func EuclideanDistance(a, b []float64) float64 {
    var sumSquares float64
    for i := range a {
        sumSquares += math.Pow(a[i] - b[i], 2)
    }
    return math.Sqrt(sumSquares)
}

func getInputBounds(input [][]float64) [][]float64 {
    maxs := make([]float64, len(input[0]))
    mins := make([]float64, len(input[0]))

    first := true
    for _, line := range input {
        for j, val := range line {
            if first {
                maxs[j] = val
                mins[j] = val
                first = false
            } else {
                if val > maxs[j] {
                    maxs[j] = val
                }
                if val < mins[j] {
                    mins[j] = val
                }
            }
        }
    }

    bounds := make([][]float64, len(mins))

    for i := range maxs {
        bounds[i] = make([]float64, 2)
        bounds[i] = []float64{mins[i], maxs[i]}
    }

    return bounds
}

func generateRandomMeans(bounds [][]float64) []float64 {
    rand.Seed(time.Now().UnixNano())
    randomMeans := make([]float64, len(bounds))

    for i := range bounds {
        randomMeans[i] = bounds[i][0] + rand.Float64() * (bounds[i][1] - bounds[i][0])
    }

    return randomMeans
}
