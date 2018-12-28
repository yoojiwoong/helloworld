from sklearn.model_selection import train_test_split
from keras.models import load_model

import numpy as np
import os

from . import Const


def load_data():
    """
    데이터를 로드하여 트레이닝, 테스트 데이터로 나눔

    Returns
    -------
    (np.array, np.array), (np.array, np.array)
        (x_train, y_train), (x_test, y_test)
    """

    file_path = os.path.join(Const.DATA_DIR, 'UCI_Credit_Card.csv')
    data = np.loadtxt(fname=file_path, delimiter=',')
    x = data[:, 0:-1]
    y = data[:, [-1]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return (x_train, y_train), (x_test, y_test)


def load_model_manager(model_name):
    """
    모델을 로드함

    Parameters
    ----------
    model_name : str
        모델의 이름(저장된 파일 이름)

    Returns
    -------
    model
        로드한 모델

    Raises
    -------
    RuntimeError
        Model 을 찾을 수 없을 때
    """

    file_path = os.path.join(Const.MODEL_DIR, model_name + '.hdf5')
    model = load_model(file_path)

    return model
