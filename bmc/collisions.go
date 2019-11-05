package bmc

import (
	"sort"

	ad "github.com/pbenner/autodiff"
	ads "github.com/pbenner/autodiff/simple"
)

// Collision is a type of collision functions
type Collision = func(
	Xs, Ps []ad.Vector,
	radius []float64,
	masses []ad.Scalar,
	numCollisions []int,
) ([]ad.Vector, []Sample, []int)

// NoCollision just resamples momenta
func NoCollision(
	Xs, Ps []ad.Vector,
	radius []float64,
	masses []ad.Scalar,
	numCollisions []int,
) ([]ad.Vector, []Sample, []int) {
	collidedSamples := make([]Sample, 0)
	for i := 0; i != len(Xs); i++ {
		convMatrix := ads.MmulS(ad.IdentityMatrix(ad.RealType, Xs[i].Dim()), masses[i])
		Ps[i] = sampleZeroMeanNormal(Xs[i].Dim(), convMatrix)
	}
	return Ps, collidedSamples, numCollisions
}

// TODO: Closest pair of points
// NormalCollision is a collision dynamics that preserves total momenta
func NormalCollision(
	Xs, Ps []ad.Vector,
	radius []float64,
	masses []ad.Scalar,
	numCollisions []int,
) ([]ad.Vector, []Sample, []int) {
	collision := make([]bool, len(Xs))
	for i := range collision {
		collision[i] = false
	}
	type collisionPair struct {
		i, j     int
		distance ad.Scalar
		dVector  ad.Vector
	}
	collisionPairList := make([]collisionPair, 0)
	for i := 0; i != len(Xs); i++ {
		for j := i + 1; j != len(Xs); j++ {
			x1, x2 := Xs[i], Xs[j]
			dVector := ads.VsubV(x1, x2)
			// Additional condition
			p1, p2 := Ps[i], Ps[j]
			deltaP := ads.VsubV(p1, p2)
			//
			distance := ads.Sqrt(ads.VdotV(dVector, dVector))
			if distance.GetValue() < radius[i]+radius[j] && ads.VdotV(dVector, deltaP).GetValue() < 0 {
				collisionPairList = append(collisionPairList, collisionPair{
					i: i, j: j,
					distance: distance,
					dVector:  dVector,
				})
			}
		}
	}
	sort.Slice(collisionPairList, func(a, b int) bool {
		return collisionPairList[a].distance.GetValue() < collisionPairList[b].distance.GetValue()
	})
	for _, pair := range collisionPairList {
		i, j := pair.i, pair.j
		if collision[i] || collision[j] {
			continue
		}
		normal := ads.VdivS(pair.dVector, pair.distance)
		p1, p2 := Ps[i], Ps[j]
		m1, m2 := masses[i], masses[j]
		avgMass := ads.Div(ads.Add(m1, m2), ad.NewReal(2))
		m2p1minusm1p2 := ads.VdotV(ads.VsubV(ads.VmulS(p1, m2), ads.VmulS(p2, m1)), normal)
		coefficient := ads.Div(m2p1minusm1p2, avgMass)
		changeOfMomentum := ads.VmulS(normal, coefficient)
		Ps[i] = ads.VaddV(p1, changeOfMomentum)
		Ps[j] = ads.VsubV(p2, changeOfMomentum)
		collision[i] = true
		collision[j] = true
		numCollisions[i]++
		numCollisions[j]++
	}
	collidedSamples := make([]Sample, 0) // optional
	for i, collide := range collision {
		if !collide {
			convMatrix := ads.MmulS(ad.IdentityMatrix(ad.RealType, Xs[i].Dim()), masses[i])
			Ps[i] = sampleZeroMeanNormal(Xs[i].Dim(), convMatrix)
		} else { // optional
			collidedSamples = append(collidedSamples, Sample{ID: i, X: Xs[i].GetValues()})
		}
	}
	return Ps, collidedSamples, numCollisions
}
