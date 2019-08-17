package cluster

import (
    "fmt"
    "math"
    "errors"
    "disgo"
    "reflect"
)

// Convert to tensor once all things are good with it
type KMeans struct {
    Input         [][]float64
    NumClusters   int
    MaxIterations int
    labels        []int
    centroids     [][]float64
}

func (k *KMeans) Optimize() [][]float64 {
    var (
        distance    float64
        minDistance float64
    )

    numsOfEachLabel := make([]float64, len(k.centroids))
    k.centroids = make([][]float64, k.NumClusters)
    k.labels = make([]int, len(k.Input))
    bounds := getInputBounds(k.Input)

    for i := 0; i < k.NumClusters; i++ {
        k.centroids[i] = make([]float64, len(bounds))
        k.centroids[i] = generateRandomMeans(bounds)
    }

    prevCentroids := make([][]float64, k.NumClusters)

    for iter := 0; iter < 1000; iter++ {
        numsOfEachLabel = make([]float64, len(k.centroids))
        for i, observation := range k.Input {
            minDistance = math.MaxFloat64
            for j, centroid := range k.centroids {
                distance = EuclideanDistance(observation, centroid)
                if distance < minDistance {
                    minDistance = distance
                    k.labels[i] = j
                }
            }
            numsOfEachLabel[k.labels[i]]++
        }

        reflect.Copy(reflect.ValueOf(prevCentroids), reflect.ValueOf(k.centroids))

        for i := range k.centroids {
            for j := range k.centroids[i] {
                k.centroids[i][j] = 0
            }
        }

        for i, label := range k.labels {
            for j := range k.centroids[label] {
                k.centroids[label][j] += k.Input[i][j] / numsOfEachLabel[label]
            }
        }

        if reflect.DeepEqual(prevCentroids, k.centroids) {
            fmt.Printf("K means clustering algorithm breaking early at %v iterations\n", iter)
            break
        }
    }

    return k.centroids
}

func (k *KMeans) Labels() []int {
    return k.labels
}

func (k *KMeans) Predict(input [][]float64) []int {
    predictions := make([]int, len(input))

    for i := 0; i < len(input); i++ {
        predictions[i] = k.Forward(input[i])
    }

    return predictions
}

func (k *KMeans) Forward(input []float64) int {
    var distance float64
    minDistance := math.MaxFloat64
    var label int
    for i := range k.centroids {
        distance = EuclideanDistance(k.centroids[i], input)
        if distance < minDistance {
            minDistance = distance
            label = i
        }
    }
    return label
}

func (KMeans) Backward() disgo.Gradient {
    err := errors.New("K Means Clustering Algorithm has no function backward")
    panic(err)
    return disgo.Gradient{}
}

// Have this input and set centroids in struct
func (KMeans) SetParameters(p disgo.Parameters) {
    err := errors.New("K Means Clustering Algorithm has no function set parameters")
    panic(err)
}
