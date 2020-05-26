package bmc

import (
	"math"
	"math/rand"

	ad "github.com/pbenner/autodiff"
	ads "github.com/pbenner/autodiff/simple"
)

// MCMC is an interface of MCMC samplers
type MCMC interface {
	Sample(
		initialX, initialP ad.Vector,
		mass ad.Scalar,
		potentialEnergy logDistribution,
		stepSize ad.Scalar,
	) (x, p ad.Vector, accepted bool, acceptance ad.Scalar)
	// getStepSize() ad.Scalar
	// setStepSize(newStepSize ad.Scalar)
}

// HMC denotes Hamiltonian Monte Carlo sampler
type HMC struct {
	StepSize ad.Scalar
	NumSteps int
}

// Sample function smaple from target distribution
func (hmc HMC) Sample(
	initialX, initialP ad.Vector,
	mass ad.Scalar,
	potentialEnergy logDistribution,
	stepSize ad.Scalar,
) (x, p ad.Vector, accepted bool, acceptance ad.Scalar) {
	if stepSize.GetValue() != 0 {
		hmc.StepSize = stepSize
	}
	H0 := hamiltonian(initialX, initialP, mass, potentialEnergy)
	x, p = clone(initialX), clone(initialP)
	for i := 0; i != hmc.NumSteps; i++ {
		x, p = leapfrog(x, p, hmc.StepSize, potentialEnergy, mass)
	}
	H := hamiltonian(x, p, mass, potentialEnergy)

	deltaH := ads.Sub(H, H0)
	if deltaH.GetValue() >= math.Log(1-rand.Float64()) {
		p = ads.VmulS(p, ad.NewReal(-1))
		accepted = true
	} else {
		x = initialX
		p = initialP
		accepted = false
	}
	acceptance = ads.Min(ad.NewReal(1), ads.Exp(ads.Neg(deltaH)))
	return x, p, accepted, acceptance
}

// NUTS denotes No-U-Turn Sampler
type NUTS struct {
	MaxDepth int
	StepSize ad.Scalar
	Delta    float64
	Depth    [][2]float64

	mass            ad.Scalar
	potentialEnergy logDistribution
}

// Sample samples from target distribution
func (nuts NUTS) Sample(
	initialX, initialP ad.Vector,
	mass ad.Scalar,
	potentialEnergy logDistribution,
	stepSize ad.Scalar,
) (x, p ad.Vector, accepted bool, acceptance ad.Scalar) {
	// Set defaults
	nuts.potentialEnergy = potentialEnergy
	nuts.Delta = 1e3
	nuts.mass = mass
	nuts.MaxDepth = 5
	if stepSize.GetValue() != 0 {
		nuts.StepSize = stepSize
	}
	x = initialX
	p = initialP

	// for adaptive step size
	var alpha ad.Scalar
	var nAlpha int

	H0 := hamiltonian(x, p, mass, potentialEnergy)
	// Sample the slice variable
	logu := ads.Add(ad.NewReal(math.Log(1-rand.Float64())), H0)

	// Initialize the tree
	xl, pl, xr, pr, depth, nelem := clone(x), clone(p), clone(x), clone(p), 0, 1.
	accepted = false
	// Integrate forward
	for {
		// Choose direction
		var dir float64
		if rand.Float64() < 0.5 {
			dir = -1
		} else {
			dir = 1
		}
		var xPrime, pPrime ad.Vector
		var nelemPrime float64
		var stop bool
		xl, pl, xr, pr, xPrime, pPrime, nelemPrime, stop, alpha, nAlpha = nuts.buildLeftOrRightTree(
			xl, pl, xr, pr, logu, dir, depth, x, p,
		)
		if stop {
			break
		}

		// Accept or reject
		if nelemPrime/nelem > rand.Float64() {
			accepted = true
			x = xPrime
			p = pPrime
		}

		nelem += nelemPrime
		depth++
		// Maximum depth of 0 (which is the default)
		// means unlimited depth
		if depth == nuts.MaxDepth {
			break
		}
	}
	nuts.updateDepth(depth)
	acceptance = ads.Div(alpha, ad.NewReal(float64(nAlpha)))
	return x, p, accepted, acceptance
}

func (nuts NUTS) buildLeftOrRightTree(
	xl, pl, xr, pr ad.Vector,
	logu ad.Scalar,
	dir float64, depth int,
	initialX, initialP ad.Vector,
) (_, _, _, _, x, p ad.Vector, nelem float64, stop bool, alpha ad.Scalar, nAlpha int) {
	if dir == -1 {
		xl, pl, _, _, x, p, nelem, stop, alpha, nAlpha = nuts.buildTree(xl, pl, logu, dir, depth, initialX, initialP)
	} else {
		_, _, xr, pr, x, p, nelem, stop, alpha, nAlpha = nuts.buildTree(xr, pr, logu, dir, depth, initialX, initialP)
	}

	if uTurn(xl, xr, pl) || uTurn(xl, xr, pr) {
		stop = true
	}
	return xl, pl, xr, pr, x, p, nelem, stop, alpha, nAlpha
}

func (nuts NUTS) buildTree(
	x, p ad.Vector,
	logu ad.Scalar,
	dir float64, depth int,
	initialX, initialP ad.Vector,
) (xl, pl, xr, pr, _, _ ad.Vector, nelem float64, stop bool, alpha ad.Scalar, nAlpha int) {
	if depth == 0 {
		// Base case: single leapfrog
		x, p := clone(x), clone(p)
		x, p = leapfrog(x, p, ads.Mul(ad.NewReal(dir), nuts.StepSize), nuts.potentialEnergy, nuts.mass)
		H1 := hamiltonian(x, p, nuts.mass, nuts.potentialEnergy)
		if H1.GetValue() >= logu.GetValue() {
			nelem = 1
		}
		if H1.GetValue()+nuts.Delta <= logu.GetValue() {
			stop = true
		}
		H0 := hamiltonian(initialX, initialP, nuts.mass, nuts.potentialEnergy)
		alpha = ads.Min(ad.NewReal(1), ads.Exp(ads.Sub(H0, H1)))
		return x, p, x, p, x, p, nelem, stop, alpha, 1
	}

	depth--
	var alphaPrime, alphaTwoPrime ad.Scalar
	var nAlphaPrime, nAlphaTwoPrime int
	xl, pl, xr, pr, x, p, nelem, stop, alphaPrime, nAlphaPrime = nuts.buildTree(x, p, logu, dir, depth, initialX, initialP)
	if stop {
		return xl, pl, xr, pr, x, p, nelem, stop, alphaPrime, nAlphaPrime
	}
	// var xPrime, pPrime ad.Vector
	// var nelem_ float64
	xl, pl, xr, pr, xPrime, pPrime, nelemPrime, stop, alphaTwoPrime, nAlphaTwoPrime := nuts.buildLeftOrRightTree(xl, pl, xr, pr, logu, dir, depth, initialX, initialP)
	nelem += nelemPrime

	// Select uniformly from nodes.
	if nelemPrime/math.Max(nelem, 1.) > rand.Float64() {
		x = xPrime
		p = pPrime
	}
	// adaptive step size
	alphaPrime = ads.Add(alphaPrime, alphaTwoPrime)
	nAlphaPrime += nAlphaTwoPrime
	return xl, pl, xr, pr, x, p, nelem, stop, alphaPrime, nAlphaPrime
}

func (nuts NUTS) updateDepth(depth int) {
	if len(nuts.Depth) <= depth {
		nuts.Depth = append(nuts.Depth,
			make([][2]float64, depth-len(nuts.Depth)+1)...)
	}
	for i := 0; i != depth; i++ {
		nuts.Depth[i][0]++
	}
	nuts.Depth[depth][1]++
}

func (nuts NUTS) getStepSize() ad.Scalar {
	return nuts.StepSize
}

func (nuts NUTS) setStepSize(newStepSize ad.Scalar) {
	nuts.StepSize = newStepSize
}
