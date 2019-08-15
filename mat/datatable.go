package mat

import (
    "strconv"
)

// Here we access with column and then row
type DataTable struct {
    VariableTypes  []VariableType
    IndexToVarName map[int]string
    VarNameToIndex map[string]int
    Values         [][]interface{}
    Target         string
}

func (t *DataTable) SetTarget(targetName string) {
    t.Target = targetName
}

func (t *DataTable) SetIntervals(intervals []string) {
    for _, variable := range intervals {
        t.VariableTypes[t.VarNameToIndex[variable]] = INTERVAL
        for j, val := range t.Values[t.VarNameToIndex[variable]] {
            conv_val, err := strconv.ParseFloat(val.(string), 64)
            if err != nil {
                panic(err)
            }
            t.Values[t.VarNameToIndex[variable]][j] = conv_val
        }
    }
}

func (t *DataTable) SetNominals(intervals []string) {
    for _, variable := range intervals {
        t.VariableTypes[t.VarNameToIndex[variable]] = NOMINAL
    }
}
