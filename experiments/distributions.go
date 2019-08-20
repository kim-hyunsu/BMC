package experiments

import (
	"math"

	ad "github.com/pbenner/autodiff"
	ads "github.com/pbenner/autodiff/simple"
)

// Distribution type denotes a type of probability density distribution
type Distribution = func(ad.Vector) ad.Scalar

// GetDistribution gets name of distribution and return corresponding function
func GetDistribution(name string) Distribution {
	switch name {
	case "AsymMOG2d":
		return AsymMOG2d
	default:
		return nil
	}
}

// AsymMOG2d is an asymmetric 2d multivariate Mixture of Gaussian (mode 3)
func AsymMOG2d(x ad.Vector) ad.Scalar {
	A := ad.NewMatrix(ad.RealType, 2, 2, []float64{0.5, 0.0, 0.0, 0.5})
	B := ad.NewMatrix(ad.RealType, 2, 2, []float64{1.0, 0.0, 0.0, 1.0})
	C := ad.NewMatrix(ad.RealType, 2, 2, []float64{2.0, 0.0, 0.0, 2.0})

	mu1 := ad.NewVector(ad.RealType, []float64{4.0 * math.Sqrt(3.0), 0.0})
	v1 := ads.VsubV(x, mu1)
	Av1 := ads.MdotV(A, v1)
	v1Av1 := ads.VdotV(v1, Av1)

	mu2 := ad.NewVector(ad.RealType, []float64{-4.0 * math.Sqrt(3.0), 0.0})
	v2 := ads.VsubV(x, mu2)
	Av2 := ads.MdotV(B, v2)
	v2Av2 := ads.VdotV(v2, Av2)

	mu3 := ad.NewVector(ad.RealType, []float64{0.0, -8.0})
	v3 := ads.VsubV(x, mu3)
	Av3 := ads.MdotV(C, v3)
	v3Av3 := ads.VdotV(v3, Av3)

	gaussian1 := ads.Exp(ads.Mul(ad.NewReal(-1), ads.Div(v1Av1, ad.NewReal(2))))
	gaussian2 := ads.Exp(ads.Mul(ad.NewReal(-1), ads.Div(v2Av2, ad.NewReal(2))))
	gaussian3 := ads.Exp(ads.Mul(ad.NewReal(-1), ads.Div(v3Av3, ad.NewReal(2))))
	MOG := ads.Div(ads.Add(ads.Add(gaussian1, gaussian2), gaussian3), ad.NewReal(3))
	return MOG
}
