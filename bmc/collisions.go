package bmc

import ad "github.com/pbenner/autodiff"

type collision = func(
	Xs, Ps []ad.Vector,
	radius float64,
	coefficients [][]map[string]ad.Scalar,
) []ad.Vector
