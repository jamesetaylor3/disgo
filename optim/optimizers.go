package optim

import (
    "fmt"
    "math"
    "disgo"
    "sync"
    "runtime"
)

const (
    VANILLA_GRADIENT_DECENT_DEFAULT_LEARNING_RATE = 1e-2
    ADAM_DEFAULT_LEARNING_RATE                    = 1e-3
    ADAM_DEFAULT_BETA_1                           = 0.9
    ADAM_DEFAULT_BETA_2                           = 0.999
    ADAM_EPSILON                                  = 1e-8
    MOMENTUM_DEFAULT_LEARNING_RATE                = 1e-3
    MOMENTUM_DEFAULT_GAMMA                        = 0.9
    RMSPROP_DEFAULT_LEARNING_RATE                 = 1e-5
    RMSPROP_DEFAULT_GAMMA                         = 0.9
)

const DEFAULT_BATCH_SIZE = 1

type Optimizer interface {
    Step(disgo.Parameters, disgo.Gradient, *sync.Mutex)
    ExportSGDHyperparameters() SGDHyperparameters
    String() string
}

type SGDHyperparameters struct {
    BatchSize     int
    NumRoutines   int
}

type VanillaGradientDescent struct {
    LearningRate  float64
    BatchSize     int
    NumRoutines   int
}

func (u *VanillaGradientDescent) Step(parameters disgo.Parameters, grad disgo.Gradient, mutex *sync.Mutex) {

    if u.LearningRate == 0 { u.LearningRate = VANILLA_GRADIENT_DECENT_DEFAULT_LEARNING_RATE }

    for i := 0; i < len(parameters); i++ {
        update := u.LearningRate * grad[i]

        mutex.Lock()
        parameters[i] -= update
        mutex.Unlock()
    }
}

func (u *VanillaGradientDescent) ExportSGDHyperparameters() SGDHyperparameters {

    if u.BatchSize == 0     { u.BatchSize = DEFAULT_BATCH_SIZE }
    if u.NumRoutines == 0   { u.NumRoutines = runtime.GOMAXPROCS(0) }

    return SGDHyperparameters{
        BatchSize: u.BatchSize,
        NumRoutines: u.NumRoutines,
    }
}

func (u *VanillaGradientDescent) String() string {
    return fmt.Sprintf("Vanilla Gradient Descent (LR: %.3e)", u.LearningRate)
}

type Adam struct {
    LearningRate  float64
    Beta_1        float64
    Beta_2        float64
    m, v          []float64
    m_hat, v_hat  []float64
    BatchSize     int
    NumRoutines   int
}

func (u *Adam) Step(parameters disgo.Parameters, grad disgo.Gradient, mutex *sync.Mutex) {

    if u.LearningRate == 0 { u.LearningRate = ADAM_DEFAULT_LEARNING_RATE }
    if u.Beta_1       == 0 { u.Beta_1 = ADAM_DEFAULT_BETA_1 }
    if u.Beta_2       == 0 { u.Beta_2 = ADAM_DEFAULT_BETA_2 }

    if len(u.m) == 0 {
        u.m = make([]float64, len(parameters))
        u.v = make([]float64, len(parameters))
        u.m_hat = make([]float64, len(parameters))
        u.v_hat = make([]float64, len(parameters))
    }

    for i := 0; i < len(parameters); i++ {
        u.m[i] = u.Beta_1 * u.m[i] + (1 - u.Beta_1) * grad[i]
        u.v[i] = u.Beta_2 * u.v[i] + (1 - u.Beta_2) * math.Pow(grad[i], 2)
        u.m_hat[i] = u.m[i] / (1 - math.Pow(u.Beta_1, float64(i + 1)))
        u.v_hat[i] = u.v[i] / (1 - math.Pow(u.Beta_2, float64(i + 1)))

        update := u.LearningRate * u.m_hat[i] / (math.Sqrt(u.v_hat[i]) + ADAM_EPSILON)
        mutex.Lock()
        parameters[i] -= update
        mutex.Unlock()

    }
}

func (u *Adam) ExportSGDHyperparameters() SGDHyperparameters {

    if u.BatchSize == 0     { u.BatchSize = DEFAULT_BATCH_SIZE }
    if u.NumRoutines == 0   { u.NumRoutines = runtime.GOMAXPROCS(0) }

    return SGDHyperparameters{
        BatchSize: u.BatchSize,
        NumRoutines: u.NumRoutines,
    }
}

func (u *Adam) String() string {
    return fmt.Sprintf("Adam (LR %.3e, B1 %.3f, B2 %.3f)", u.LearningRate, u.Beta_1, u.Beta_2)
}

type Momentum struct {
    LearningRate  float64
    Gamma         float64
    v             []float64
    BatchSize     int
    NumRoutines   int
}

func (u *Momentum) Step(parameters disgo.Parameters, grad disgo.Gradient, mutex *sync.Mutex) {

    if u.LearningRate == 0 { u.LearningRate = MOMENTUM_DEFAULT_LEARNING_RATE }
    if u.Gamma        == 0 { u.Gamma = MOMENTUM_DEFAULT_GAMMA }

    if len(u.v) == 0 {
        u.v = make([]float64, len(parameters))
    }

    for i := 0; i < len(parameters); i++ {
        u.v[i] = u.Gamma * u.v[i] + u.LearningRate * grad[i]
        mutex.Lock()
        parameters[i] -= u.v[i]
        mutex.Unlock()
    }
}

func (u *Momentum) ExportSGDHyperparameters() SGDHyperparameters {

    if u.BatchSize == 0     { u.BatchSize = DEFAULT_BATCH_SIZE }
    if u.NumRoutines == 0   { u.NumRoutines = runtime.GOMAXPROCS(0) }

    return SGDHyperparameters{
        BatchSize: u.BatchSize,
        NumRoutines: u.NumRoutines,
    }
}

func (u *Momentum) String() string {
    return fmt.Sprintf("Momentum (LR %.3e, Gamma %.3f)", u.LearningRate, u.Gamma)
}

type RMSProp struct {
    LearningRate  float64
    Gamma         float64
    v             []float64
    BatchSize     int
    NumRoutines   int
}

func (u *RMSProp) Step(parameters disgo.Parameters, grad disgo.Gradient, mutex *sync.Mutex) {

    if u.LearningRate == 0 { u.LearningRate = RMSPROP_DEFAULT_LEARNING_RATE }
    if u.Gamma        == 0 { u.Gamma = RMSPROP_DEFAULT_GAMMA }

    if len(u.v) == 0 {
        u.v = make([]float64, len(parameters))
    }

    for i := 0; i < len(parameters); i++ {
        u.v[i] = u.Gamma * u.v[i] + (1 - u.Gamma) * math.Pow(grad[i], 2)
        update := grad[i] * u.LearningRate / math.Sqrt(u.v[i])
        mutex.Lock()
        parameters[i] -= update
        mutex.Unlock()
    }
}

func (u *RMSProp) ExportSGDHyperparameters() SGDHyperparameters {

    if u.BatchSize == 0     { u.BatchSize = DEFAULT_BATCH_SIZE }
    if u.NumRoutines == 0   { u.NumRoutines = runtime.GOMAXPROCS(0) }

    return SGDHyperparameters{
        BatchSize: u.BatchSize,
        NumRoutines: u.NumRoutines,
    }
}

func (u *RMSProp) String() string {
    return fmt.Sprintf("RMSProp (LR %.3e, Gamma %.3f)", u.LearningRate, u.Gamma)
}
