// Maybe different filename and placement


package io

import (
    "os"
    "io/ioutil"
    "encoding/json"
    "disgo"
    "disgo/linearmodels"
)

// This whole pacakge only really works with linear models
// It might have to become one with interfaces{}s bc when you think of all the diffrent stuff with nn's decision trees etc, its gonna be rlly hard hahaha

type modeljson struct {
    ModelType  string           `json:"ModelType"`
    Bias       bool             `json:Bias`
    Parameters disgo.Parameters `json:Parameters`
}

func LoadModelFromJSON(path string) func([]float64) float64 {
    jsonFile, err := os.Open(path)

    if err != nil {
        panic(err)
    }
    defer jsonFile.Close()

    byteValue, _ := ioutil.ReadAll(jsonFile)

    var model_json modeljson
    var model func([]float64) float64

    json.Unmarshal(byteValue, &model_json)

    switch model_json.ModelType {
    case "linearmodels.LogisticRegressionModel":
        model = (&linearmodels.LogisticRegressionModel{
            Bias: model_json.Bias,
            Parameters: model_json.Parameters,
        }).Forward

    case "linearmodels.LinearRegressionModel":
        model = (&linearmodels.LinearRegressionModel{
            Bias: model_json.Bias,
            Parameters: model_json.Parameters,
        }).Forward

    }

    return model
}

// could have this defined in the
func ExportModelToJSON(model disgo.Model, path string) {

}
