package minimize

import (
	"disgo"
	"fmt"
	"time"
	"disgo/optim"
)

const (
	H_DERIVATIVE           = 1e-10
	DEFAULT_PRECISION      = 1e-8
	DEFAULT_MAX_ITERATIONS = 100000
)

type Minimizer struct {
	Function          func(disgo.Parameters) float64
	GradientFunc      func(disgo.Parameters) disgo.Gradient
	Optimizer 		  optim.Optimizer
	Dimension         int
	InitialParameters disgo.Parameters
	MaxIterations     int
	Precision         float64
	Loud              bool // Maybe only keep this option for testing
}

// Maybe make this pointer but bug if so in linear regression <------- loook
func (o Minimizer) Run() disgo.Parameters {

	start := time.Now()

	if o.GradientFunc == nil { o.GradientFunc = NumericalGradient(o.Function) }
	if o.Dimension == 0 { o.Dimension = 1 }
	if o.Precision == 0 { o.Precision = DEFAULT_PRECISION }
	if o.MaxIterations == 0 { o.MaxIterations = DEFAULT_MAX_ITERATIONS }

	current_parameters := make(disgo.Parameters, o.Dimension)
	grad := make(disgo.Gradient, o.Dimension)

	if len(o.InitialParameters) != 0 { current_parameters = o.InitialParameters }

	var i int
	for ; i < o.MaxIterations; i++ {
		grad = o.GradientFunc(current_parameters)

		o.Optimizer.Step(current_parameters, grad, nil)

	}

	if o.Loud {
		t := time.Now()
		elapsed := t.Sub(start)
		fmt.Printf("\nTook Gradient Descent %v iterations and %s\n", i, elapsed)
	}

	return current_parameters

}

func NumericalGradient(function func(disgo.Parameters) float64) func(disgo.Parameters) disgo.Gradient {
	return func(parameters disgo.Parameters) disgo.Gradient {
		//return num_grad(function, p)
		var top, bottom, slope float64

		grad := make(disgo.Gradient, len(parameters))
		parameters_H := make(disgo.Parameters, len(parameters))
		copy(parameters_H, parameters)

		// This whole indexes business for the prev_i is a mess--consider changing please
		parameters_H[len(parameters)-1] += H_DERIVATIVE

		for i, _ := range parameters {
			parameters_H[(i+len(parameters)-1)%len(parameters)] -= H_DERIVATIVE
			parameters_H[i] += H_DERIVATIVE

			top = function(parameters_H) - function(parameters)
			bottom = H_DERIVATIVE
			slope = top / bottom
			grad[i] = slope

		}
		return grad
	}
}
