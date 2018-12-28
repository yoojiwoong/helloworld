# helloworld

# UCI Credit Problem
url : https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
## Requirements
* Python 3.5 (or later)
* TensorFlow
* Keras
* scikit-learn
## Methods for Training and Prediction
* ```nn```
* ```dnn```
* ```lstm```
## How to use
* Train Data and Save Model
```shell
python main.py --act make --algo [method] --model [modelname]
``` * Evaluate Model
```shell
python main.py --act eval --algo [method] --model [modelname]
``` ## Examples
* Train Data and Save Model
```shell
python main.py --act "make" --algo "nn" --model "nn_model"
``` * Evaluate Model
```shell
python main.py --act "eval" --algo "nn" --model "nn_model"
