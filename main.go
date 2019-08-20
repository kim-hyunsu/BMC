package main

import (
	"flag"
	"fmt"
	"math/rand"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/kim-hyunsu/BrownianMonteCarlo/experiments"

	"github.com/kim-hyunsu/BrownianMonteCarlo/bmc"
	ad "github.com/pbenner/autodiff"
)

func main() {
	numParticles := flag.Int("numParticles", runtime.NumCPU(), "Number of particles.")
	numSamples := flag.Int("numSamples", 1000, "Number of samples per particle.")
	numSteps := flag.Int("numSteps", 10, "Number of steps (L).")
	stepSize := flag.Float64("stepSize", 0.1, "Size of a step (epsilon)")
	collision := flag.String("collision", "NormalCollision", "Type of collision.")
	mcmc := flag.String("mcmc", "nuts", "HMC or NUTS.")
	radius := flag.Float64("radius", 1.0, "Radius of each particle.")
	mass := flag.String("mass", "1.0", "Masses of each particle")
	dist := flag.String("dist", "", "Target probability distribution.")

	flag.Parse()

	var sampler bmc.MCMC
	var collide bmc.Collision

	// Sampler
	switch *mcmc {
	case "nuts":
		sampler = bmc.NUTS{
			StepSize: ad.NewScalar(ad.RealType, *stepSize),
		}
	case "hmc":
		sampler = bmc.HMC{
			StepSize: ad.NewScalar(ad.RealType, *stepSize),
			NumSteps: *numSteps,
		}
	default:
		panic("No Sampler.")
	}

	// Collide
	switch *collision {
	case "NormalCollision":
		collide = bmc.NormalCollision
	default:
		collide = bmc.NoCollision
	}
	masses := make([]ad.Scalar, *numParticles)
	for i, m := range strings.Split(*mass, ",") {
		mass, _ := strconv.ParseFloat(m, 64)
		masses[i] = ad.NewScalar(ad.RealType, mass)
	}
	sample := make(chan bmc.Sample)
	collidedSample := make(chan bmc.Sample, *numSamples)

	rand.Seed(time.Now().UTC().UnixNano())
	BMC := bmc.BrownianMonteCarlo{
		Sampler:      sampler,
		Collide:      collide,
		NumParticles: *numParticles,
		Radius:       *radius,
		Masses:       masses,
	}
	target := experiments.GetDistribution(*dist)
	begin := time.Now()
	BMC.Sample(target, ad.NewVector(ad.RealType, []float64{0., 0.}), sample, collidedSample)
	samples := make([]bmc.Sample, 0)
	for i := 0; i != *numSamples; i++ {
		samples = append(samples, <-sample)
	}
	BMC.Stop()
	fmt.Println("[", time.Since(begin), "]")
	collidedSamples := make([]bmc.Sample, 0)
	for s := range collidedSample {
		collidedSamples = append(collidedSamples, s)
	}

	experiments.PlotScatters(
		BMC,
		samples,
		*numParticles, *numSamples,
		*radius,
		BMC.NumAccepted, BMC.NumRejected, BMC.NumCollisions,
		target,
	)
}
