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
	var A ad.Matrix
	if covariance.Equals(ad.IdentityMatrix(ad.RealType, dim), 0.0001) {
		A = covariance
	} else {
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
		A = ad.NewMatrix(ad.RealType, dim, dim, buffer)
	}
	AZ := ads.MdotV(A, Z)
	return AZ
}

func kineticEnergy(momentum ad.Vector, mass ad.Scalar) ad.Scalar {
	inverseMassMomentum := ads.VdivS(momentum, mass)
	return ads.Mul(ad.NewReal(0.5), ads.VdotV(momentum, inverseMassMomentum))
}

func hamiltonian(position, momentum ad.Vector, mass ad.Scalar, potentialEnergy logDistribution) ad.Scalar {
	return ads.Add(potentialEnergy(position), kineticEnergy(momentum, mass))
}

func gradients(f logDistribution, x ad.Vector) ad.Vector {
	x.Variables(1)
	s := f(x)
	gradients := make([]float64, x.Dim())
	for i := 0; i < x.Dim(); i++ {
		gradients[i] = s.GetDerivative(i)
	}
	return ad.NewVector(ad.RealType, gradients)
}

func leapfrog(position, momentum ad.Vector, stepSize ad.Scalar, potentialEnergy logDistribution, mass ad.Scalar) (ad.Vector, ad.Vector) {
	grad := gradients(potentialEnergy, position)
	momentumChange := ads.VmulS(grad, ads.Mul(ad.NewReal(0.5), stepSize))
	p := ads.VsubV(momentum, momentumChange)
	x := ads.VaddV(position, ads.VmulS(ads.VdivS(p, mass), stepSize))
	grad = gradients(potentialEnergy, x)
	momentumChange = ads.VmulS(grad, ads.Mul(ad.NewReal(0.5), stepSize))
	p = ads.VsubV(p, momentumChange)
	return x, p
}

func uTurn(xl, xr, p ad.Vector) bool {
	dot := ads.VdotV(ads.VsubV(xr, xl), p)
	return dot.GetValue() < 0
}

// Float64ToVector converts float64 to autodiff.Vector
func Float64ToVector(arr []float64) ad.Vector {
	return ad.NewVector(ad.RealType, arr)
}

// VectorToFloat64 converts autodiff.Vector to float64
func VectorToFloat64(vec ad.Vector) []float64 {
	return vec.GetValues()
}

func updateRadius(
	oldRadius float64,
	newPotential, oldPotential ad.Scalar,
	S ad.Scalar,
) (newRadius float64) {
	oldRadiusScalar := ad.NewReal(oldRadius)
	newRadiusScalar := ads.Add(oldRadiusScalar, ads.Div(ads.Sub(newPotential, oldPotential), S))
	newRadius = newRadiusScalar.GetValue()
	return newRadius
}
