"""Нейронная сеть с тремя входами и одним выходом. Обучение методом градиентного спуска."""

weights = [0.1, 0.2, -0.1]  # Начальные веса

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]  # Данные для входа
win_or_lose_binary = [1, 1, 0, 1]  # Реальные результаты

input = [toes[0], wlrec[0], nfans[0]]  # Точки входа
true = win_or_lose_binary[0]  # То, каким должно быть предсказание
alpha = 0.01  # Альфа- коэффициент


def w_sum(a, b):  # Функция, вычисляющая скалярное произведение входа на веса
    assert(len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
        return output


def neural_network(input, weights):  # Функция нейронной сети(по сути в данном примере ненужная)
    pred = w_sum(input, weights)
    return pred


pred = neural_network(input, weights)  # Вычисляем пресказание, подавая в функцию нужные данные
error = (pred - true) ** 2  # Вычисляем среднеквадратическую ошибку
delta = pred - true  # Вычисляем чистую ошибку


def ele_mul(number, vector):  # Функция, вычисляющая 3 приращения для каждого начального веса. На вход подаётся одна \
    # дельта и вектор из входных данных. Дельта перемножается на каждый из входов, получая 3 разных числа, на которые\
    # нужно изменить первоначальные веса
    output = [0, 0, 0]
    assert(len(output) == len(vector))
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output


weight_deltas = ele_mul(delta, input)  # Получаем вектор из приращений(чисел, на которые нужно будет менять веса)


for i in range(len(weights)):  # Функция, меняющая веса на соответствующие числа из weight_deltas
    weights[i] -= alpha * weight_deltas[i]


print(f"Weights: {weights}\nWeight Deltas: {weight_deltas}")


