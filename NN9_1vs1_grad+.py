"""Нейронная сеть с одним входом и одним выходом. Обучение методом градиентного спуска. Более сложное предсказание."""
############################################################

weight = 0.0
alpha = 0.01  # Фиксируется перед обучением. В разы уменьшает шаг изменения веса для того, чтобы найти как можно более \
# точное значение.
number_of_toes = [1.1]
win_or_lose_binary = [0.8]
input = number_of_toes[0]
goal_pred = win_or_lose_binary[0]


def neural_network(input, weight):
    prediction = input * weight
    return prediction


for i in range(10000):
    print(f"{i+1}------\nWeight: {weight}")
    pred = neural_network(input, weight)
    error = (pred - goal_pred) ** 2

    delta = pred - goal_pred  # Чистая ошибка(не среднеквадратическая). Разность между предсказанием и реальным исходом

    weight_delta = input * delta  # Шаг изменения веса. То, на сколько нужно будет менять вес при градиентном спуске,
    # но только чистая ошибка записана как delta

    weight -= weight_delta * alpha  # alpha в разы уменьшает шаг изменения веса для того, чтобы найти как можно более
    # \точное значение.
    print(f"Error: {error}    Prediction: {pred}\nDelta: {delta}    Weight Delta: {weight_delta}")
    i += 1
