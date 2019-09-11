package experiments

import (
	"bufio"
	"encoding/csv"
	"os"
	"strconv"
	"strings"

	"github.com/kim-hyunsu/BrownianMonteCarlo/bmc"
)

// GetNameFromBMC composes a filename to save
func GetNameFromBMC(BMC bmc.BrownianMonteCarlo, collsion string, target string, numSamples int) string {
	var samplerName string
	switch BMC.Sampler.(type) {
	case bmc.HMC:
		samplerName = "HMC"
	case bmc.NUTS:
		samplerName = "NUTS"
	default:
		samplerName = "UndefinedSampler"
	}
	collsionName := collsion
	valueofParticles := strconv.Itoa(BMC.NumParticles)
	valueofRadius := strconv.FormatFloat(BMC.Radius, 'f', -1, 64)
	valueofSamples := strconv.Itoa(numSamples)
	targetName := target
	valueofParticles = strings.Join([]string{"P", valueofParticles}, "")
	valueofRadius = strings.Join([]string{"R", valueofRadius}, "")
	valueofSamples = strings.Join([]string{"S", valueofSamples}, "")
	filename := strings.Join([]string{samplerName, collsionName, targetName, valueofParticles, valueofRadius, valueofSamples}, "_")
	return filename
}

// ToCSV creates a file storing sample data
func ToCSV(path string, samples []bmc.Sample, BMC bmc.BrownianMonteCarlo) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	wr := csv.NewWriter(bufio.NewWriter(file))
	for _, s := range samples {
		id := strconv.Itoa(s.ID)
		mass := strconv.FormatFloat(BMC.Masses[s.ID].GetValue(), 'f', -1, 64)
		collision := strconv.Itoa(BMC.NumCollisions[s.ID])
		accepted := strconv.Itoa(BMC.NumAccepted[s.ID])
		rejected := strconv.Itoa(BMC.NumRejected[s.ID])
		vector := make([]string, 0)
		for _, v := range s.X {
			vector = append(vector, strconv.FormatFloat(v, 'f', -1, 64))
		}
		line := append([]string{id, mass, collision, accepted, rejected}, vector...)
		wr.Write(line)
	}
	wr.Flush()
	return nil
}
