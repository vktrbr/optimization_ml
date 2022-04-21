package main

import "C"
import (
	"fmt"
	"math"
)

func main() {
	// fmt.Println("Hello world!")

	fmt.Println(Derivative(testFunc, [2]float64{1., 2.}, 0.001))

}

//export doSome
func doSome() {
	println("Hello from Go")
}

//export getSomeNumber
func getSomeNumber() int8 {
	return 5
}

//export factorial
func factorial(n int64) int64 {
	var fact int64 = 1
	var i int64
	for i = 1; i <= n; i++ {
		fact *= i
	}
	return fact
}

// export Derivative
// This function calculate derivative of function at point x0
func Derivative(function func([2]float64) float64, x0 [2]float64, h float64) float64 {
	if h == 0. {
		h = 1e-4
	}
	// x0_len = len(x0)
	x0[0] = x0[0]
	val := function(x0)
	return val
}

func testFunc(x [2]float64) float64 {
	return math.Pow(x[0], 2) + math.Pow(x[1], 2)
}
