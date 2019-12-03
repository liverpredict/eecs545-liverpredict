#!/bin/zsh
python main.py --x_train_file data/x_train.csv --x_test_file data/x_test.csv --y_train_file data/y_180_train.csv --y_test_file data/y_180_test.csv --epochs 50 --num_hidden_layers 3 --dropout 0.4

python main.py --x_train_file data/x_train.csv --x_test_file data/x_test.csv --y_train_file data/y_180_train.csv --y_test_file data/y_180_test.csv --epochs 50 --num_hidden_layers 4 --dropout 0.4

python main.py --x_train_file data/x_train.csv --x_test_file data/x_test.csv --y_train_file data/y_180_train.csv --y_test_file data/y_180_test.csv --epochs 50 --num_hidden_layers 5 --dropout 0.4

python main.py --x_train_file data/x_train.csv --x_test_file data/x_test.csv --y_train_file data/y_365_train.csv --y_test_file data/y_365_test.csv --epochs 50 --num_hidden_layers 3 --dropout 0.4

python main.py --x_train_file data/x_train.csv --x_test_file data/x_test.csv --y_train_file data/y_365_train.csv --y_test_file data/y_365_test.csv --epochs 50 --num_hidden_layers 4 --dropout 0.4

python main.py --x_train_file data/x_train.csv --x_test_file data/x_test.csv --y_train_file data/y_365_train.csv --y_test_file data/y_365_test.csv --epochs 50 --num_hidden_layers 5 --dropout 0.4

python main.py --x_train_file data/x_train.csv --x_test_file data/x_test.csv --y_train_file data/y_730_train.csv --y_test_file data/y_730_test.csv --epochs 50 --num_hidden_layers 3 --dropout 0.4

python main.py --x_train_file data/x_train.csv --x_test_file data/x_test.csv --y_train_file data/y_730_train.csv --y_test_file data/y_730_test.csv --epochs 50 --num_hidden_layers 4 --dropout 0.4

python main.py --x_train_file data/x_train.csv --x_test_file data/x_test.csv --y_train_file data/y_730_train.csv --y_test_file data/y_730_test.csv --epochs 50 --num_hidden_layers 5 --dropout 0.4

python main.py --x_train_file data/x_180_train_oversample.csv --x_test_file data/x_test.csv --y_train_file data/y_180_train_oversample.csv --y_test_file data/y_180_test.csv --epochs 50 --num_hidden_layers 3 --dropout 0.4

python main.py --x_train_file data/x_180_train_oversample.csv --x_test_file data/x_test.csv --y_train_file data/y_180_train_oversample.csv --y_test_file data/y_180_test.csv --epochs 50 --num_hidden_layers 4 --dropout 0.4

python main.py --x_train_file data/x_180_train_oversample.csv --x_test_file data/x_test.csv --y_train_file data/y_180_train_oversample.csv --y_test_file data/y_180_test.csv --epochs 50 --num_hidden_layers 5 --dropout 0.4

python main.py --x_train_file data/x_365_train_oversample.csv --x_test_file data/x_test.csv --y_train_file data/y_365_train_oversample.csv --y_test_file data/y_365_test.csv --epochs 50 --num_hidden_layers 3 --dropout 0.4

python main.py --x_train_file data/x_365_train_oversample.csv --x_test_file data/x_test.csv --y_train_file data/y_365_train_oversample.csv --y_test_file data/y_365_test.csv --epochs 50 --num_hidden_layers 4 --dropout 0.4

python main.py --x_train_file data/x_365_train_oversample.csv --x_test_file data/x_test.csv --y_train_file data/y_365_train_oversample.csv --y_test_file data/y_365_test.csv --epochs 50 --num_hidden_layers 5 --dropout 0.4

python main.py --x_train_file data/x_730_train_oversample.csv --x_test_file data/x_test.csv --y_train_file data/y_730_train_oversample.csv --y_test_file data/y_730_test.csv --epochs 50 --num_hidden_layers 3 --dropout 0.4

python main.py --x_train_file data/x_730_train_oversample.csv --x_test_file data/x_test.csv --y_train_file data/y_730_train_oversample.csv --y_test_file data/y_730_test.csv --epochs 50 --num_hidden_layers 4 --dropout 0.4

python main.py --x_train_file data/x_730_train_oversample.csv --x_test_file data/x_test.csv --y_train_file data/y_730_train_oversample.csv --y_test_file data/y_730_test.csv --epochs 50 --num_hidden_layers 5 --dropout 0.4