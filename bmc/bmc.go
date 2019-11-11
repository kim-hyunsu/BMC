package bmc

import (
	"fmt"
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

	// Adaptive step size
	dualAvgVarList []map[string]ad.Scalar
	count          int
	MaxAdapt       int
	Delta          ad.Scalar
}

// Sample samples a vector from target distribution(dist) and put it in a channel(sample)
func (bmc *BrownianMonteCarlo) Sample(
	dist distribution,
	initialX ad.Vector,
	sample, collidedSample chan Sample,
) {
	// Initialize
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
	S := ad.NewReal(50 * bmc.Radius[0])
	bmc.InitialRadius = bmc.Radius[0]

	// Initialize sampler
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
		// adaptive step size
		bmc.dualAvgVarList[i] = map[string]ad.Scalar{
			"eps":    bmc.findReasonableEpsilon(initialX),
			"epsBar": ad.NewScalar(ad.RealType, 1),
			"HBar":   ad.NewScalar(ad.RealType, 0),
		}
	}
	for i := 0; i != bmc.NumParticles; i++ {
		bmc.Radius[i] = updateRadius(bmc.Radius[i], potentials[i], maxPotential, S)
	}

	// Adaptive step size
	mu := ads.Log(ads.Mul(ad.NewReal(10), bmc.dualAvgVarList[0]["eps"]))
	gamma := ad.NewScalar(ad.RealType, 0.05)
	t0 := ad.NewScalar(ad.RealType, 10)
	kappa := ad.NewScalar(ad.RealType, 0.75)
	if bmc.Delta == nil {
		bmc.Delta = ad.NewScalar(ad.RealType, 0.5)
	}

	// Sampling (parallelized)
	go func() {
		defer close(sample)
		// defer close(collidedSample)
		for {
			if bmc.stop {
				break
			}
			bmc.count++
			done := make(chan bool, bmc.NumParticles)
			for i := 0; i != bmc.NumParticles; i++ {
				go func(id int) {
					x, p, accepted, acceptance := bmc.Sampler.Sample(
						Xs[id], Ps[id], bmc.Masses[id], bmc.potentialEnergy, bmc.dualAvgVarList[id]["eps"],
					)
					if accepted {
						bmc.NumAccepted[id]++
					} else {
						bmc.NumRejected[id]++
					}
					Xs[id], Ps[id] = x, p

					// adaptive radius
					newPotential := bmc.potentialEnergy(Xs[id])
					bmc.Radius[id] = updateRadius(bmc.Radius[id], newPotential, potentials[id], S)
					potentials[id] = newPotential

					// adaptive step size
					bmc.dualAvgVarList[id]["acceptance"] = acceptance

					sample <- Sample{ID: id, X: x.GetValues()}
					done <- true
				}(i)
			}
			for i := 0; i != bmc.NumParticles; i++ {
				<-done
			}
			// fmt.Println(bmc.Radius)
			Ps, _, bmc.NumCollisions = bmc.Collide(Xs, Ps, bmc.Radius, bmc.Masses, bmc.NumCollisions)

			// Adaptive step size
			bmc.dualAveraging(mu, gamma, t0, kappa, bmc.Delta)
			fmt.Println(bmc.dualAvgVarList)
		}
	}()
}

func (bmc *BrownianMonteCarlo) dualAveraging(mu, gamma, t0, kappa, delta ad.Scalar) {
	if bmc.count <= bmc.MaxAdapt {
		m := ad.NewScalar(ad.RealType, float64(bmc.count))
		temp := ads.Div(ad.NewReal(1), ads.Add(m, t0))
		for i := 0; i != bmc.NumParticles; i++ {
			HBar := bmc.dualAvgVarList[i]["Hbar"]
			acceptance := bmc.dualAvgVarList[i]["acceptance"]
			epsBar := bmc.dualAvgVarList[i]["epsBar"]
			HBar = ads.Add(ads.Mul(HBar, ads.Sub(ad.NewReal(1), temp)), ads.Mul(temp, ads.Sub(delta, acceptance)))
			logEps := ads.Sub(mu, ads.Mul(HBar, ads.Div(ads.Sqrt(m), gamma)))
			logEpsBar := ads.Add(ads.Mul(logEps, ads.Pow(m, ads.Neg(kappa))), ads.Mul(ads.Log(epsBar), ads.Sub(ad.NewReal(1), ads.Pow(m, ads.Neg(kappa)))))
			bmc.dualAvgVarList[i]["HBar"] = HBar
			bmc.dualAvgVarList[i]["epsBar"] = ads.Exp(logEpsBar)
			bmc.dualAvgVarList[i]["eps"] = ads.Exp(logEps)
		}
	}
}

func (bmc *BrownianMonteCarlo) findReasonableEpsilon(x ad.Vector) (eps ad.Scalar) {
	eps = ad.NewScalar(ad.RealType, 1)
	p := sampleZeroMeanNormal(x.Dim(), ad.IdentityMatrix(ad.RealType, x.Dim()))
	xPrime, pPrime := leapfrog(x, p, eps, bmc.potentialEnergy, bmc.Masses[0])
	var a ad.Scalar
	H1 := hamiltonian(xPrime, pPrime, bmc.Masses[0], bmc.potentialEnergy)
	H0 := hamiltonian(x, p, bmc.Masses[0], bmc.potentialEnergy)
	ratio := ads.Exp(ads.Sub(H0, H1))
	if ratio.GetValue() > 0.5 {
		a = ad.NewScalar(ad.RealType, 1)
	} else {
		a = ad.NewScalar(ad.RealType, -1)
	}
	for ads.Pow(ratio, a).GetValue() > ads.Pow(ad.NewReal(2), ads.Neg(a)).GetValue() {
		eps = ads.Mul(eps, ads.Pow(ad.NewReal(2), a))
		xPrime, pPrime = leapfrog(xPrime, pPrime, eps, bmc.potentialEnergy, bmc.Masses[0])
	}
	return eps
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
