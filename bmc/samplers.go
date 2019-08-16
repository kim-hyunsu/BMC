package bmc

import ad "github.com/pbenner/autodiff"

// MCMC is an interface of MCMC samplers
type MCMC interface {
	Sample() (x, p ad.Vector, accepted bool)
}
