go run main.go -numParticles=$1 -numSamples=$2 -dist=$3 -dim=$4 -radius=$5 -stepSize=$6 -mcmc=$7 -collision=$8
path="csv/$7_$8_$3_P$1_R$5_S$2.csv"
python3 plot-samples.py ${path}
python3 plot-histogram.py ${path}
python3 plot-moment.py ${path}
