package bmc

import (
	"math/rand"

	ad "github.com/pbenner/autodiff"
	ads "github.com/pbenner/autodiff/simple"
	"gonum.org/v1/gonum/mat"
)

type distribution = func(ad.Vector) ad.Scalar
type logDistribution = func(ad.Vector) ad.Scalar

func minusLogDist(dist distribution) logDistribution {
	return func(x ad.Vector) ad.Scalar {
		return ads.Mul(ad.NewReal(-1), ads.Log(dist(x)))
	}
}

func calculateCollisionCoefficients(masses []ad.Scalar) [][]map[string]ad.Scalar {
	return nil
}

func clone(x ad.Vector) ad.Vector {
	_x := make([]float64, x.Dim())
	copy(_x, x.GetValues())
	return ad.NewVector(ad.RealType, _x)
}

// sampleZeroMeanNormal samples a vector from multivariate normal distribution whose mean is zero.
func sampleZeroMeanNormal(dim int, covariance ad.Matrix) ad.Vector {
	sample := make([]float64, dim)
	for i := range sample {
		sample[i] = rand.NormFloat64()
	}
	Z := ad.NewVector(ad.RealType, sample)
	var chol mat.Cholesky
	Sigma := mat.NewSymDense(dim, covariance.GetValues())
	ok := chol.Factorize(Sigma)
	if !ok {
		panic("Can't generate multivariate normal distribution")
	}
	buffer := make([]float64, 0)
	for i := 0; i != dim; i++ {
		for j := 0; j != dim; j++ {
			buffer = append(buffer, chol.At(i, j))
		}
	}
	A := ad.NewMatrix(ad.RealType, dim, dim, buffer)
	AZ := ads.MdotV(A, Z)
	return AZ
}
