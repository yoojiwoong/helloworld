from .. import Manager


def make_model(model_name, input_shape, batch_size, num_epochs, x_train, y_train):
    """
    DNN 기법으로 데이터를 학습함.

    Parameters
    ----------
    model_name : str
        모델의 이름(저장되는 파일 이름)
    input_shape : shape
        input data 의 shape
    batch_size : int
        batch 크기
    num_epochs : int
        epoch 횟수
    x_train : np.array
        train 데이터의 입력
    y_train : np.array
        train 데이터의 출력

    Returns
    -------
    model, history
        생성된 모델과 학습 과정을 기록한 history
    """

    model, history = []

    return model, history


def evaluate(model_name, x_test, y_test):
    """
    DNN 기법으로 평가함

    Parameters
    ----------
    model_name : str
        모델의 이름(저장되는 파일 이름)
    x_test : np.array
        test 데이터의 입력
    y_test : np.array
        test 데이터의 출력

    Returns
    -------
    np.array, acc
        모델이 예측한 값들, 정확도

    Raises
    -------
    RuntimeError
        Model 을 찾을 수 없을 때
    """

    model = Manager.load_model_manager(model_name)
    pred, acc = []

    return pred, acc


