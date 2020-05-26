package main

import (
	"flag"
	"fmt"
	"runtime"
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
	stepSize := flag.Float64("stepSize", 0., "Size of a step (epsilon)")
	collision := flag.String("collision", "NormalCollision", "Type of collision.")
	mcmc := flag.String("mcmc", "nuts", "HMC or NUTS.")
	radius := flag.Float64("radius", 1.0, "Radius of each particle.")
	mass := flag.Float64("mass", 1.0, "Masses of each particle")
	dist := flag.String("dist", "", "Target probability distribution.")
	dim := flag.Int("dim", 2, "Dimension of target distribution.")
	verbose := flag.Bool("verbose", false, "List all samples")

	flag.Parse()

	var sampler bmc.MCMC
	var collide bmc.Collision
	var maxAdapt int

	// Sampler
	switch *mcmc {
	case "NUTS":
		sampler = bmc.NUTS{
			StepSize: ad.NewScalar(ad.RealType, *stepSize),
		}
	case "HMC":
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
	radii := make([]float64, *numParticles)
	for i := 0; i != *numParticles; i++ {
		// masses[i] = ad.NewScalar(ad.RealType, *mass)
		masses[i] = ad.NewScalar(ad.RealType, (*mass)*float64(i+1))
		radii[i] = *radius
	}
	sample := make(chan bmc.Sample)
	collidedSample := make(chan bmc.Sample, *numSamples)

	// adaptive step size
	if *stepSize == 0. {
		maxAdapt = *numSamples / 100
	} else {
		maxAdapt = 0
	}
	BMC := bmc.BrownianMonteCarlo{
		Sampler:      sampler,
		Collide:      collide,
		NumParticles: *numParticles,
		Radius:       radii,
		Masses:       masses,
		MaxAdapt:     maxAdapt,
	}
	target := experiments.GetDistribution(*dist)
	initialX := make([]float64, *dim)
	begin := time.Now()
	BMC.Sample(target, ad.NewVector(ad.RealType, initialX), sample, collidedSample)
	samples := make([]bmc.Sample, 0)
	for i := 0; i != *numSamples; i++ {
		s := <-sample
		if *verbose {
			fmt.Println(i, s)
		}
		samples = append(samples, s)
	}
	fmt.Println("[", time.Since(begin), "]")
	BMC.Stop()
	// collidedSamples := make([]bmc.Sample, 0)
	// for s := range collidedSample {
	// 	collidedSamples = append(collidedSamples, s)
	// }

	// experiments.PlotScatters(
	// 	BMC,
	// 	samples,
	// 	*numParticles, *numSamples,
	// 	*radius,
	// 	BMC.NumAccepted, BMC.NumRejected, BMC.NumCollisions,
	// 	target,
	// 	*collision,
	// )
	filename := experiments.GetNameFromBMC(BMC, *collision, *dist, len(samples))
	path := strings.Join([]string{"csv/", filename, ".csv"}, "")
	err := experiments.ToCSV(path, samples, BMC)
	if err != nil {
		panic(err)
	}
}
