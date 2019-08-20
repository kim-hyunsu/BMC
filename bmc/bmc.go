package bmc

import (
	"time"

	ad "github.com/pbenner/autodiff"
)

// Sample is a structure that receives sampled value
type Sample struct {
	ID int
	X  []float64
}

// BrownianMonteCarlo simulates collisions of particles.
type BrownianMonteCarlo struct {
	// Hyperparameter
	Sampler      MCMC
	Collide      Collision
	NumParticles int
	Radius       float64
	Masses       []ad.Scalar

	// Statistics
	NumCollisions []int
	NumAccepted   []int
	NumRejected   []int

	// Private attributes
	sample          chan Sample
	collidedSample  chan Sample
	coefficients    [][]map[string]ad.Scalar
	stop            bool
	potentialEnergy logDistribution
}

// Sample samples a vector from target distribution(dist) and put it in a channel(sample)
func (bmc *BrownianMonteCarlo) Sample(
	dist distribution,
	intialX ad.Vector,
	sample, collidedSample chan Sample,
) {
	bmc.sample = sample
	bmc.collidedSample = collidedSample
	bmc.potentialEnergy = minusLogDist(dist)
	bmc.NumAccepted = make([]int, bmc.NumParticles)
	bmc.NumRejected = make([]int, bmc.NumParticles)
	bmc.NumCollisions = make([]int, bmc.NumParticles)
	bmc.coefficients = calculateCollisionCoefficients(bmc.Masses)

	Xs := make([]ad.Vector, bmc.NumParticles)
	Ps := make([]ad.Vector, bmc.NumParticles)
	for i := 0; i != bmc.NumParticles; i++ {
		Xs[i] = clone(intialX)
		variance := bmc.Masses[i].GetValue()
		Ps[i] = sampleZeroMeanNormal(intialX.Dim(), ad.NewMatrix(ad.RealType, 2, 2, []float64{
			variance, 0,
			0, variance,
		}))
		if bmc.NumAccepted[i] != 0 { // REMOVABLE(for assertion)
			panic("bmc's statistics are not zero-initialized.")
		}
	}
	go func() {
		defer close(sample)
		defer close(collidedSample)
		for {
			if bmc.stop {
				break
			}
			done := make(chan bool, bmc.NumParticles)
			for i := 0; i != bmc.NumParticles; i++ {
				go func(id int) {
					x, p, accepted := bmc.Sampler.Sample(
						Xs[id], Ps[id], bmc.Masses[id], bmc.potentialEnergy,
					)
					if accepted {
						bmc.NumAccepted[id]++
					} else {
						bmc.NumRejected[id]++
					}
					Xs[id], Ps[id] = x, p
					sample <- Sample{ID: id, X: x.GetValues()}
					done <- true
				}(i)
			}
			for i := 0; i != bmc.NumParticles; i++ {
				<-done
			}
			Ps, _, bmc.NumCollisions = bmc.Collide(Xs, Ps, bmc.Radius, bmc.Masses, bmc.NumCollisions)
		}
	}()
}

// Stop stops sampling
func (bmc *BrownianMonteCarlo) Stop() {
	bmc.stop = true
	for {
		select {
		case _, ok := <-bmc.sample:
			if !ok {
				return
			}
		case _, ok := <-bmc.collidedSample:
			if !ok {
				return
			}
		default:
			time.Sleep(1000)
		}
	}
}

// Mass returns mass of a particle
func (bmc *BrownianMonteCarlo) Mass(id int) ad.Scalar {
	return bmc.Masses[id]
}
