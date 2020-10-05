"""Нейронная сеть с одним входом и несколькими выходами"""
############################################################

weights = [0.3, 0.2, 0.9]  # Веса
wlrec = [0.65, 0.8, 0.8, 0.9]  # Статистика, из которой берётся точка данных для входа

input = wlrec[0]  # Точка данных на вход


def ele_mul(number, vector):
    output = [0, 0, 0]  # Вектор(список) данных на выходе
    assert(len(output) == len(vector))  # Проверка количества элементов весов и количества элементов на выходе
    for i in range(len(vector)):
        output[i] = number * vector[i]  # Получение выходных данных с помощью умножения входа на весы
    return output  # Результат- вектор выходных данных


def neural_network(input, weights):
    pred = ele_mul(input, weights)  # Передаём в функцию нужные входные данные и веса
    return pred  # Результат- вектор предсказаний


pred = neural_network(input, weights)

print(pred)






