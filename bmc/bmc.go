package bmc

import (
	"time"

	ad "github.com/pbenner/autodiff"
	ads "github.com/pbenner/autodiff/simple"
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
	Radius       []float64
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

	// For plotting
	InitialRadius float64
}

// Sample samples a vector from target distribution(dist) and put it in a channel(sample)
func (bmc *BrownianMonteCarlo) Sample(
	dist distribution,
	initialX ad.Vector,
	sample, collidedSample chan Sample,
) {
	bmc.sample = sample
	bmc.collidedSample = collidedSample
	bmc.potentialEnergy = minusLogDist(dist)
	bmc.NumAccepted = make([]int, bmc.NumParticles)
	bmc.NumRejected = make([]int, bmc.NumParticles)
	bmc.NumCollisions = make([]int, bmc.NumParticles)
	// bmc.coefficients = calculateCollisionCoefficients(bmc.Masses)  // legacy

	// Adaptive radius
	potentials := make([]ad.Scalar, bmc.NumParticles)
	maxPotential := ad.NewScalar(ad.RealType, -9999)
	epsilon := ad.NewReal(50 * bmc.Radius[0])
	bmc.InitialRadius = bmc.Radius[0]

	Xs := make([]ad.Vector, bmc.NumParticles)
	Ps := make([]ad.Vector, bmc.NumParticles)
	for i := 0; i != bmc.NumParticles; i++ {
		// initial X
		Xs[i] = clone(initialX)
		// initial P
		convMatrix := ads.MmulS(ad.IdentityMatrix(ad.RealType, initialX.Dim()), bmc.Masses[i])
		Ps[i] = sampleZeroMeanNormal(initialX.Dim(), convMatrix)
		// current potential energies
		potentials[i] = bmc.potentialEnergy(Xs[i])
		// find max potential
		if potentials[i].GetValue() > maxPotential.GetValue() {
			maxPotential = potentials[i]
		}
	}
	for i := 0; i != bmc.NumParticles; i++ {
		bmc.Radius[i] = updateRadius(bmc.Radius[i], potentials[i], maxPotential, epsilon)
	}
	go func() {
		defer close(sample)
		// defer close(collidedSample)
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
					newPotential := bmc.potentialEnergy(Xs[id])
					bmc.Radius[id] = updateRadius(bmc.Radius[id], newPotential, potentials[id], epsilon)
					potentials[id] = newPotential
					sample <- Sample{ID: id, X: x.GetValues()}
					done <- true
				}(i)
			}
			for i := 0; i != bmc.NumParticles; i++ {
				<-done
			}
			// fmt.Println(bmc.Radius)
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
