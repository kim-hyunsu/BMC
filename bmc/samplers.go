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
	) (x, p ad.Vector, accepted bool)
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
) (x, p ad.Vector, accepted bool) {
	H0 := hamiltonian(initialX, initialP, mass, potentialEnergy)
	x, p = clone(initialX), clone(initialP)
	for i := 0; i != hmc.NumSteps; i++ {
		x, p = leapfrog(x, p, hmc.StepSize, potentialEnergy, mass)
	}
	H := hamiltonian(x, p, mass, potentialEnergy)

	if ads.Sub(H, H0).GetValue() >= math.Log(1-rand.Float64()) {
		p = ads.VmulS(p, ad.NewReal(-1))
		accepted = true
	} else {
		x = initialX
		p = initialP
		accepted = false
	}
	return x, p, accepted
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
) (x, p ad.Vector, accepted bool) {
	// Set defaults
	nuts.potentialEnergy = potentialEnergy
	nuts.Delta = 1e3
	nuts.mass = mass
	nuts.MaxDepth = 10
	x = initialX
	p = initialP

	H0 := hamiltonian(x, p, mass, potentialEnergy)
	// Sample the slice variable
	logu := ads.Add(ad.NewReal(math.Log(1-rand.Float64())), H0)

	// Initialize the tree
	xl, pl, xr, pr, depth, nelem := x, p, x, p, 0, 1.
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
		xl, pl, xr, pr, xPrime, pPrime, nelemPrime, stop = nuts.buildLeftOrRightTree(
			xl, pl, xr, pr, logu, dir, depth,
		)
		if stop {
			break
		}

		// Accept or reject
		if nelemPrime/nelem > rand.Float64() {
			accepted = true
			x = xPrime
			p = ads.VmulS(pPrime, ad.NewReal(-1))
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
	return x, p, accepted
}

func (nuts NUTS) buildLeftOrRightTree(
	xl, pl, xr, pr ad.Vector,
	logu ad.Scalar,
	dir float64, depth int,
) (_, _, _, _, x, p ad.Vector, nelem float64, stop bool) {
	if dir == -1 {
		xl, pl, _, _, x, p, nelem, stop = nuts.buildTree(xl, pl, logu, dir, depth)
	} else {
		_, _, xr, pr, x, p, nelem, stop = nuts.buildTree(xr, pr, logu, dir, depth)
	}

	if uTurn(xl, xr, pl) || uTurn(xl, xr, pr) {
		stop = true
	}
	return xl, pl, xr, pr, x, p, nelem, stop
}

func (nuts NUTS) buildTree(
	x, p ad.Vector,
	logu ad.Scalar,
	dir float64, depth int,
) (xl, pl, xr, pr, _, _ ad.Vector, nelem float64, stop bool) {
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

		return x, p, x, p, x, p, nelem, stop
	}

	depth--

	xl, pl, xr, pr, x, p, nelem, stop = nuts.buildTree(x, p, logu, dir, depth)
	if stop {
		return xl, pl, xr, pr, x, p, nelem, stop
	}
	// var xPrime, pPrime ad.Vector
	// var nelem_ float64
	xl, pl, xr, pr, xPrime, pPrime, nelemPrime, stop := nuts.buildLeftOrRightTree(xl, pl, xr, pr, logu, dir, depth)
	nelem += nelemPrime

	// Select uniformly from nodes.
	if nelemPrime/math.Max(nelem, 1.) > rand.Float64() {
		x = xPrime
		p = pPrime
	}
	return xl, pl, xr, pr, x, p, nelem, stop
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
