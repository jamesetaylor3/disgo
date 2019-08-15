package mat

type VariableType int

const (
    UNKNOWN VariableType = iota
    NOMINAL
    INTERVAL
    ORDINAL
    DATETIME
    TARGET
)

func (vartype VariableType) String() string {
    names := [...]string{
        "UNKNOWN",
        "NOMINAL",
        "INTERVAL",
        "ORDINAL",
        "DATETIME",
    }

    if vartype > DATETIME {
        return "UNKNOWN"
    }

    return names[vartype]
}
