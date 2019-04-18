# different scale factors
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.25 --p_crx 0.9 --tour_size 5
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.55 --p_crx 0.9 --tour_size 5
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.9 --tour_size 5
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.95 --p_crx 0.9 --tour_size 5
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 1.20 --p_crx 0.9 --tour_size 5

# different crossover prob
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.1 --tour_size 5
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.3 --tour_size 5
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.5 --tour_size 5
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.7 --tour_size 5
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.9 --tour_size 5

# different tournament size
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.9 --tour_size 4
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.9 --tour_size 5
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.9 --tour_size 6
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.9 --tour_size 7
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.9 --tour_size 8

# different population size
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.9 --pop_size 10 --tour_size 4
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.9 --pop_size 20 --tour_size 4
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.9 --pop_size 30 --tour_size 4
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.9 --pop_size 40 --tour_size 4
python mnist_DE.py --epochs 50 --seed 4 --scale_factor 0.75 --p_crx 0.9 --pop_size 50 --tour_size 4