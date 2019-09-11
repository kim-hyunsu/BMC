package experiments

import (
	"github.com/kim-hyunsu/BrownianMonteCarlo/bmc"
	ad "github.com/pbenner/autodiff"
	ads "github.com/pbenner/autodiff/simple"
)

// KLDivKNN gives KL-divergence via k-nearest-neighbor distance
func KLDivKNN(samples []bmc.Sample, dist Distribution) float64 {
	return 0.
}

func getMean(samples [][]float64) []float64 {
	length := len(samples)
	dim := len(samples[0])
	sum := make([]float64, dim)
	for _, s := range samples {
		for i := 0; i != dim; i++ {
			sum[i] += s[i] / float64(length)
		}
	}
	return sum
}

// ESS denotes effective sample size. cf) https://jwalton.info/Efficient-effective-sample-size-python/
func ESS(m int, samples []bmc.Sample, dist Distribution) float64 {
	sampleList := make([][][]float64, m)
	for _, s := range samples {
		sampleList[s.ID] = append(sampleList[s.ID], s.X)
	}
	n := len(samples) / m
	cumAutocorr := ad.NewScalar(ad.RealType, 0)
	autocorr := ad.NewScalar(ad.RealType, 0)
	t := 0
	for autocorr.GetValue() >= 0 {
		// Variogram
		sum := ad.NewScalar(ad.RealType, 0)
		for j := 0; j != m; j++ {
			for i := t; i != n; i++ {
				ij, imtj := bmc.Float64ToVector(sampleList[j][i]), bmc.Float64ToVector(sampleList[j][i-t])
				distance := ads.Sub(dist(ij), dist(imtj))
				distSquare := ads.Mul(distance, distance)
				sum = ads.Add(sum, distSquare)
			}
		}
		Vt := ads.Div(sum, ad.NewReal(float64(m*(n-t-1))))
		// Within-sequence variance
		sum = ad.NewScalar(ad.RealType, 0)
		for j := 0; j != m; j++ {
			meanj := ad.NewScalar(ad.RealType, 0)
			for i := 0; i != n; i++ {
				ij := dist(bmc.Float64ToVector(sampleList[j][i]))
				meanj = ads.Add(meanj, ij)
			}
			meanj = ads.Div(meanj, ad.NewReal(float64(n)))
			for i := 0; i != n; i++ {
				ij := dist(bmc.Float64ToVector(sampleList[j][i]))
				distance := ads.Sub(ij, meanj)
				distSquare := ads.Mul(distance, distance)
				sum = ads.Add(sum, distSquare)
			}
		}
		W := ads.Div(sum, ad.NewReal(float64(m*(n-1))))
		// TODO Between-sequence variance
		mean := ad.NewScalar(ad.RealType, 0)
		for j := 0; j != m; j++ {
			meanj := ad.NewScalar(ad.RealType, 0)
			for i := 0; i != n; i++ {
				ij := dist(bmc.Float64ToVector(sampleList[j][i]))
				meanj = ads.Add(meanj, ij)
			}
			meanj = ads.Div(meanj, ad.NewReal(float64(n)))
			mean = ads.Add(mean, meanj)
		}
		mean = ads.Div(mean, ad.NewReal(float64(m)))
		sum = ad.NewScalar(ad.RealType, 0)
		for j := 0; j != m; j++ {
			meanj := ad.NewScalar(ad.RealType, 0)
			for i := 0; i != n; i++ {
				ij := dist(bmc.Float64ToVector(sampleList[j][i]))
				meanj = ads.Add(meanj, ij)
			}
			meanj = ads.Div(meanj, ad.NewReal(float64(n)))
			distance := ads.Sub(meanj, mean)
			distSquare := ads.Mul(distance, distance)
			sum = ads.Add(sum, distSquare)
		}
		B := ads.Mul(ad.NewReal(float64(n/(m-1))), sum)
		varPlus := ads.Add(ads.Mul(ad.NewReal(float64((n-1)/n)), W), ads.Div(B, ad.NewReal(float64(n))))
		autocorr = ads.Sub(ad.NewReal(1), ads.Div(ads.Div(Vt, varPlus), ad.NewReal(2)))
		cumAutocorr = ads.Add(cumAutocorr, autocorr)
		t++
	}
	return float64(m*n) / (1. + 2*cumAutocorr.GetValue())
}
