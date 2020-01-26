package cluster

import (
    "fmt"
    "math"
    "errors"
    "reflect"
)

// Probably place kernel elsewhere in own pacakge?
type Kernel func (float64) float64

func Gaussian(sigma float64) Kernel {
    if (sigma <= 0) {
        err := errors.New("Gaussian kernel takes a sigma that must be greater than 0!")
        panic(err)
    }
    return func(x float64) float64 {
        return math.Exp(-x / (2 * math.Pow(sigma, 2)))
    }
}

func Flat(lambda float64) Kernel {
    if (lambda <= 0) {
        err := errors.New("Flat kernel takes a lambda that must be greater than 0!")
        panic(err)
    }
    return func(x float64) float64 {
        if x <= lambda {
            return 1
        } else {
            return 0
        }
    }
}

const PRECISION float64 = 1e-3

type MeanShift struct {
    Kernel
    NumWindows int
    labels     []int
    centroids  [][]float64
}

func (k *MeanShift) Fit(input [][]float64, maxIterations int) [][]float64 {
    if k.NumWindows == 0 {
        err := errors.New("Cannot fit k means! NumWindows is either \"0\" or improperly defined!")
        panic(err)
    }

    var (
        sumWeight float64
        distance  float64
    )

    k.centroids = make([][]float64, k.NumWindows)
    k.labels = make([]int, len(input))
    observationWeights := make([]float64, len(input))
    bounds := getInputBounds(input)

    for i := 0; i < k.NumWindows; i++ {
        k.centroids[i] = make([]float64, len(bounds))
        k.centroids[i] = generateRandomMeans(bounds)
    }

    prevCentroids := make([][]float64, k.NumWindows)

    for iter := 0; iter < maxIterations; iter++ {
        for i, centroid := range k.centroids {
            for j, observation := range input {
                distance = EuclideanDistance(observation, centroid)
                observationWeights[j] = k.Kernel(distance)
                sumWeight += observationWeights[j]
            }

            for j, weight := range observationWeights {
                for l := range centroid {
                    k.centroids[i][l] += (weight / sumWeight) * input[j][l]
                }
            }
        }

        for i := range k.centroids {
            copy(prevCentroids[i], k.centroids[i])
        }

        if reflect.DeepEqual(prevCentroids, k.centroids) {
            fmt.Printf("Meanshift clustering algorithm breaking early at %v iterations\n", iter)
            break
        }
    }

    fmt.Println(k.centroids)

    // Reduce the number of centroids
    // Label each point to a centroid

    return k.centroids
}

func (k *MeanShift) Labels() []int {
    return k.labels
}

func (k *MeanShift) Predict(input [][]float64) []int{
    predictions := make([]int, len(input))

    for i := 0; i < len(input); i++ {
        predictions[i] = k.Forward(input[i])
    }

    return predictions
}

func (k *MeanShift) Forward(input []float64) int {
    var (
        distance    float64
        minDistance float64
        label       int
    )
    minDistance = math.MaxFloat64

    for i := range k.centroids {
        distance = EuclideanDistance(k.centroids[i], input)
        if distance < minDistance {
            minDistance = distance
            label = i
        }
    }
    return label
}
