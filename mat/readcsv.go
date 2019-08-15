package mat

import (
	"encoding/csv"
	"fmt"
	"os"
	"io"
	"regexp"
	"strconv"
	"time"
)


func DataTableFromCSV(path string, row_num, col_num int, header bool) *DataTable {
    start := time.Now()

    dt := &DataTable{}
	dt.VariableTypes = make([]VariableType, col_num + 1)
    dt.IndexToVarName = map[int]string{}
    dt.VarNameToIndex = map[string]int{}

	csv_data := make([][]interface{}, col_num + 1)
	for i := range csv_data {
		csv_data[i] = make([]interface{}, row_num)
	}

	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

    r := csv.NewReader(f)

    var line []string
    var i int
    for {
        line, err = r.Read()

        if err == io.EOF {
            break
        }
        /** if err != nil {
            panic(err)
        }**/

        if header {
            for j, val := range line {
                dt.IndexToVarName[j] = val
                dt.VarNameToIndex[val] = j
            }
            header = false
            continue
        }
		for j, val := range line {
			csv_data[j][i] = val
		}
        i++
	}

	t := time.Now()
	elapsed := t.Sub(start)

	fmt.Printf("readcsv loaded %v (%s)\n", path, elapsed)

    dt.Values = csv_data

	return dt
}


// For [][]float64
func sanatizedata(in string) string {
	reg, err := regexp.Compile("[^0-9.]")
	if err != nil {
		panic(err)
	}
	out := reg.ReplaceAllString(in, "")

	return out

}

func ReadCSV(path string, row_num, col_num int) [][]float64 {

	start := time.Now()

	csv_data := make([][]float64, row_num)
	for i := range csv_data {
		csv_data[i] = make([]float64, col_num)
	}

	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	lines, err := csv.NewReader(f).ReadAll()
	if err != nil {
		panic(err)
	}

	for i, line := range lines {
		for j, val := range line {
			val = sanatizedata(val)
			conv_data, err := strconv.ParseFloat(val, 64)
			if err != nil {
				panic(err)
			}
			csv_data[i][j] = conv_data
		}
	}

	t := time.Now()
	elapsed := t.Sub(start)

	fmt.Printf("readcsv loaded %v (%s)\n", path, elapsed)

	return csv_data

}
