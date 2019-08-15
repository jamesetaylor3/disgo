package stat

import (
    "math"
)

// This is dirty how all the functins call each other
// Include correlation and stuff

func Mean(nums []float64) float64 {
    var sum float64
    for _, num := range nums {
        sum += num
    }
    return sum / float64(len(nums))
}

func Variance(nums []float64) float64 {
    var sum float64
    mean := Mean(nums)

    for _, num := range nums {
        sum += math.Pow(num - mean, 2)
    }

    return sum / float64(len(nums) - 1)
}

func StandardDeviation(nums []float64) float64 {
    return math.Sqrt(Variance(nums))
}

func Kurtosis(nums []float64) float64 {
    var kurtosis float64
    mean := Mean(nums)
    std := StandardDeviation(nums)

    for _, num := range nums {
        kurtosis += math.Pow((num - mean) / std, 4) / float64(len(nums))
    }

    return kurtosis
}

func Skewness(nums []float64) float64 {
    var skewness float64
    mean := Mean(nums)
    std := StandardDeviation(nums)

    for _, num := range nums {
        skewness += math.Pow((num - mean) / std, 3) / float64(len(nums))
    }

    return skewness
}
