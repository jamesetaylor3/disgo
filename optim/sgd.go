package optim

import (
	"disgo"
	"disgo/mat"
	"fmt"
	"time"
	"sync"
)

type SGD struct {
	Optimizer
	Model     disgo.Model
	Dimension int
	Loud      bool
	Input     [][]float64
	Target    []float64
}

func (o SGD) Run() disgo.Parameters {

	var (
		batchCounter        int
		totalIterations     int
		wg                  sync.WaitGroup
	    counterMutex        sync.Mutex
		parametersMutex     sync.Mutex
	)

	// add a batch counter that is set ++ and moduloed by len(o.Input) / p.BatchSize blah blah

	p := o.Optimizer.ExportSGDHyperparameters()

	start := time.Now()

	current_parameters := make(disgo.Parameters, o.Dimension)
	inputLength := len(o.Input)

	wg.Add(p.NumRoutines)

	for r := 0; r < p.NumRoutines; r++ {
		go func() {
			grad := make(disgo.Gradient, o.Dimension)

			for {
				grad = o.Model.Backward(o.Input[batchCounter*p.BatchSize:(batchCounter+1)*p.BatchSize], o.Target[batchCounter*p.BatchSize:(batchCounter+1)*p.BatchSize])

				o.Optimizer.Step(current_parameters, grad, &parametersMutex)


				o.Model.SetParameters(current_parameters)

				counterMutex.Lock()
				totalIterations++
				batchCounter = (batchCounter + 1) % (inputLength / p.BatchSize)
				counterMutex.Unlock()

				if o.Loud && (totalIterations % (p.MaxIterations / 10) == 0) {
					// Print precision when we get it to work to act as a performance of the model
					fmt.Printf("Iteration: %8.f, Time %s\n", float64(totalIterations), time.Now().Sub(start))
				}

				if batchCounter == 0 {
					mat.ShuffleWithTarget(o.Input, o.Target)
				}

				// have a precision break as well

				if totalIterations >= p.MaxIterations {
					wg.Done()
					break
				}
			}
		}()
	}

	wg.Wait()

	if o.Loud {
		t := time.Now()
		elapsed := t.Sub(start)
		fmt.Printf("\nTook Stochastic Gradient Descent with %s %v iterations with batch sizes of %v on %v goroutines in %s\n", o.Optimizer, totalIterations, p.BatchSize, p.NumRoutines, elapsed)
	}

	return current_parameters
}
