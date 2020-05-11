# disgo

This library was primarly meant to teach myself the inner workings of many machine learning data structures and algorithms. It is not stable at all and has a lot of dead code / functionality. I plan on updating is as i learn more and more about certain techniques. Here are some sample applications.

### Logistic regression

```go
import (
	...
	"disgo/mat"
	"disgo/optim"
	"disgo/linearmodels"
	"disgo/metrics"
)

...

// binary classification data for breast cancer
table := mat.ReadCSV("breast_cancer.csv")

train_table, test_table := mat.SampleData(table, 0.5)

train_input, train_target := mat.ExtractTarget(train_table)

log_reg := linearmodels.LogisticRegressionModel{
	Criterion: metrics.CrossEntropyLoss,
	Optimizer: &optim.RMSProp{
		LearningRate: 1e-5,
		BatchSize:    50,
		NumRoutines:  1,
	},
}

_ = log_reg.Fit(train_input, train_target, 100000)

test_pred := log_reg.Predict(test_input)
evaluation := metrics.EvaluationClassification(test_target, test_pred)

fmt.Printf("Cross Entropy Loss: %.3f\n", evaluation["CrossEntropyLoss"])
fmt.Printf("Classification Accuracy: %.1f%%\n", 100 * evaluation["ClassificationAccuracy"])

```

### Function minimization

Note that the minimize.Booth and minimize.Banana variables are simply functions that represent the Banana and Booth functions. minimize.G is a gradient function for `f(x) = pow(x - 2, 2)`

```go
import (
	...
	"disgo"
	"disgo/optim"
	"disgo/minimize"
)

him_optim := minimize.Minimizer{
	Function:          minimize.Booth,
	Dimension:         2,
	InitialParameters: disgo.Parameters{0, -0},
	Optimizer: &optim.Momentum{
		LearningRate: 1e-2,
		Gamma:        0.95,
	},
	Loud: true,
}
fmt.Printf("himmeleblau local min: %.5f\n", him_optim.Run())

f_optim := minimize.Minimizer{
	GradientFunc: minimize.G,
	Optimizer: &optim.Momentum{
		LearningRate: 1e-2,
		Gamma:        0.95,
	},
	Loud: true,
}
fmt.Printf("f local min: %.5f\n", f_optim.Run())

ban_optim := minimize.Minimizer{
	Function:  minimize.Banana,
	Dimension: 2,
	Optimizer: &optim.Adam{
		LearningRate: 1e-4,
		Beta_1:       0.9,
		Beta_2:       0.999,
	},
	Loud: true,
}
fmt.Printf("banana local min: %.5f\n", ban_optim.Run())
...
