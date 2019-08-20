package experiments

import (
	"fmt"
	"image/color"
	"reflect"
	"runtime"
	"strconv"
	"strings"

	"github.com/kim-hyunsu/BrownianMonteCarlo/bmc"
	ad "github.com/pbenner/autodiff"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func pointColor(id int, numParticles int) color.RGBA {
	ratio := float64(id) / float64(numParticles+1)
	return color.RGBA{
		R: uint8(255 * ratio),
		G: uint8(225 * (1 - ratio)),
		B: uint8(255 * ((id + 1) % 2)),
		A: 255,
	}
}

func getFunctionName(i interface{}) string {
	name := runtime.FuncForPC(reflect.ValueOf(i).Pointer()).Name()
	splitted := strings.Split(name, ".")
	return splitted[len(splitted)-1]
}

func getImageName(path string, sampler bmc.MCMC, function interface{}, numParticles int, numSamples int, radius float64) string {
	base := strings.Join([]string{path, getFunctionName(function)}, "")
	var samplerName string
	switch sampler.(type) {
	case bmc.HMC:
		samplerName = "HMC"
	case bmc.NUTS:
		samplerName = "NUTS"
	default:
		samplerName = "UndefinedSampler"
	}
	particles := strconv.Itoa(numParticles)
	samples := strconv.Itoa(numSamples)
	Radius := strconv.FormatFloat(radius, 'f', -1, 64)
	particles = strings.Join([]string{particles, "particles"}, "")
	samples = strings.Join([]string{samples, "samples"}, "")
	Radius = strings.Join([]string{Radius, "radius"}, "")
	name := strings.Join([]string{base, samplerName, particles, samples, Radius}, "_")
	return strings.Join([]string{name, ".png"}, "")
}

// PlotScatters plot data scatters
func PlotScatters(
	BMC bmc.BrownianMonteCarlo,
	samples []bmc.Sample,
	numParticles, numSamples int,
	radius float64,
	numAccepted, numRejected, numCollision []int,
	targetDistribution func(ad.Vector) ad.Scalar,
) {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Add(plotter.NewGrid())
	for i := 0; i != numParticles; i++ {
		numAccepted := float64(numAccepted[i])
		numRejected := float64(numRejected[i])
		acceptPercent := 100 * numAccepted / (numAccepted + numRejected)
		fmt.Println(BMC.Mass(i), acceptPercent, "% Accepted,", numCollision[i], "collided.")
		data := make(plotter.XYs, len(samples))
		for j, x := range samples {
			if x.ID == i {
				data[j].X = x.X[0]
				data[j].Y = x.X[1]
			}
		}
		if radius != 0 {
			p.Title.Text = getImageName("bmc-results/", BMC.Sampler, targetDistribution, numParticles, numSamples, radius)
		} else {
			p.Title.Text = getImageName("multi-hmc-results/", BMC.Sampler, targetDistribution, numParticles, numSamples, radius)
		}
		s, err := plotter.NewScatter(data)
		if err != nil {
			panic(err)
		}
		s.GlyphStyle.Shape = draw.CircleGlyph{}
		s.GlyphStyle.Radius = vg.Points(1)
		s.GlyphStyle.Color = pointColor(i, numParticles)
		p.Add(s)
	}
	if err := p.Save(4*vg.Inch, 4*vg.Inch, p.Title.Text); err != nil {
		panic(err)
	}
}
