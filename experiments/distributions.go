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
	case "AsymMOG10d":
		return AsymMOG10d
	case "Sym16GM2d":
		return Sym16GM2d
	default:
		return nil
	}
}

// AsymMOG2d is an asymmetric 2d multivariate Mixture of Gaussian (mode 3)
func AsymMOG2d(x ad.Vector) ad.Scalar {
	A := ad.NewMatrix(ad.RealType, 2, 2, []float64{0.5, 0.0, 0.0, 0.5})
	B := ad.NewMatrix(ad.RealType, 2, 2, []float64{1.0, 0.0, 0.0, 1.0})
	C := ad.NewMatrix(ad.RealType, 2, 2, []float64{2.0, 0.0, 0.0, 2.0})

	mu1 := ad.NewVector(ad.RealType, []float64{4.0*math.Sqrt(3.0) + 1., 1.})
	v1 := ads.VsubV(x, mu1)
	Av1 := ads.MdotV(A, v1)
	v1Av1 := ads.VdotV(v1, Av1)

	mu2 := ad.NewVector(ad.RealType, []float64{-4.0*math.Sqrt(3.0) - 1., 1.})
	v2 := ads.VsubV(x, mu2)
	Av2 := ads.MdotV(B, v2)
	v2Av2 := ads.VdotV(v2, Av2)

	mu3 := ad.NewVector(ad.RealType, []float64{3., -11.})
	v3 := ads.VsubV(x, mu3)
	Av3 := ads.MdotV(C, v3)
	v3Av3 := ads.VdotV(v3, Av3)

	gaussian1 := ads.Exp(ads.Mul(ad.NewReal(-1), ads.Div(v1Av1, ad.NewReal(2))))
	gaussian2 := ads.Exp(ads.Mul(ad.NewReal(-1), ads.Div(v2Av2, ad.NewReal(2))))
	gaussian3 := ads.Exp(ads.Mul(ad.NewReal(-1), ads.Div(v3Av3, ad.NewReal(2))))
	MOG := ads.Div(ads.Add(ads.Add(gaussian1, gaussian2), gaussian3), ad.NewReal(3))
	return MOG
}

// Sym16GM2d is an symmetric 2d multivariate Gaussian mixture (mode 16)
func Sym16GM2d(x ad.Vector) ad.Scalar {
	modes := 16
	GM := ad.NewScalar(ad.RealType, 0)
	for i := 0; i != modes; i++ {
		A := ad.NewMatrix(ad.RealType, 2, 2, []float64{1., 0., 0., 1.})
		width := int(math.Sqrt(float64(modes)))
		a, b := float64(i/width), float64(i%width)
		mu := ad.NewVector(ad.RealType, []float64{a * 10, b * 10})
		v := ads.VsubV(x, mu)
		Av := ads.MdotV(A, v)
		vAv := ads.VdotV(v, Av)

		gaussian := ads.Exp(ads.Mul(ad.NewReal(-1), ads.Div(vAv, ad.NewReal(2))))
		GM = ads.Add(GM, gaussian)
	}
	return ads.Div(GM, ad.NewReal(float64(modes)))
}

// AsymMOG10d is an asymmetric 100d multivariate Mixture of Gaussian (mode 3)
func AsymMOG10d(x ad.Vector) ad.Scalar {
	dim := 10
	MOG := ad.NewScalar(ad.RealType, 0)
	for i := 0; i != 3; i++ {
		eye := ad.IdentityMatrix(ad.RealType, dim)
		// Minv := ads.MmulS(eye, ad.NewReal(1*math.Pow(2, float64(i))))
		Minv := eye
		rawMu := make([]float64, 0)
		for j := 0; j != dim; j++ {
			rawMu = append(rawMu, float64(i)*5)
			// rawMu = append(rawMu, math.Pow(-1, float64(i))*2.+math.Pow(-1, float64(j))*float64(i+1))
		}
		mu := ad.NewVector(ad.RealType, rawMu)
		V := ads.VsubV(x, mu)
		MinvV := ads.MdotV(Minv, V)
		VMinvV := ads.VdotV(V, MinvV)

		gaussian := ads.Exp(ads.Mul(ad.NewReal(-1), ads.Div(VMinvV, ad.NewReal(2))))
		MOG = ads.Add(MOG, gaussian)
	}
	return ads.Div(MOG, ad.NewReal(3))
}
