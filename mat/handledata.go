package mat

import (
    "fmt"
	"math/rand"
    "time"
)

// Assuming target data is last column
// Make it so you can set column name or fucking index or whatever
func ExtractTarget(csv_data [][]float64) ([][]float64, []float64) {

	start := time.Now()

	input := make([][]float64, len(csv_data))
	for i := range input {
		input[i] = make([]float64, len(csv_data[0])-1)
	}
	target := make([]float64, len(csv_data))

	for i, line := range csv_data {
		input[i] = line[0 : len(line)-1]
		target[i] = line[len(line)-1]
	}

	t := time.Now()
	elapsed := t.Sub(start)

	fmt.Printf("splitdata split target from input (%s)\n", elapsed)

	return input, target
}

func ShuffleData(in [][]float64) {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(in), func(i, j int) { in[i], in[j] = in[j], in[i] })
}


// Rename this jawn
func ShuffleWithTarget(in [][]float64, t []float64) {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(in), func(i, j int) {
		in[i], in[j] = in[j], in[i]
		t[i], t[j] = t[j], t[i]
	})
}

func SampleData(in [][]float64, pct float64) ([][]float64, [][]float64) {
	start := time.Now()

	partition_loc := int(pct * float64(len(in)))

	sample := in[0:partition_loc]
	outOfSample := in[partition_loc:]

	t := time.Now()
	elapsed := t.Sub(start)
	fmt.Printf("data partitioned into two sub-tables (%s)\n", elapsed)

	return sample, outOfSample
}
