from classifier import *
from classifier import algorithms
import argparse

if __name__ == "__main__":
    # Argument Setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--act', required=True, type=str, choices=['make', 'eval'])
    parser.add_argument('--algo', required=True, type=str, choices=['nn', 'dnn', 'lstm'])
    parser.add_argument('--model', required=True, type=str)

    args = parser.parse_args()
    act_name = args.act
    algorithm_name = args.algo
    model_name = args.model

    # Data
    (x_train, y_train), (x_test, y_test) = Manager.load_data()
    input_shape = x_train[0].shape

    # Action
    if act_name == 'make':
        if algorithm_name == 'nn':
            algorithms.nn.make_model(model_name, input_shape, batch_size=32, num_epochs=5, x_train=x_train, y_train=y_train)
        elif algorithm_name == 'dnn':
            algorithms.dnn.make_model(model_name, input_shape, batch_size=32, num_epochs=5, x_train=x_train, y_train=y_train)
        elif algorithm_name == 'lstm':
            algorithms.lstm.make_model(model_name, input_shape, batch_size=32, num_epochs=5, x_train=x_train, y_train=y_train)
    else:
        if algorithm_name == 'nn':
            algorithms.nn.evaluate(model_name, x_test, y_test)
        elif algorithm_name == 'dnn':
            algorithms.dnn.evaluate(model_name, x_test, y_test)
        elif algorithm_name == 'lstm':
            algorithms.lstm.evaluate(model_name, x_test, y_test)

