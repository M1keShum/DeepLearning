"""Нейронная сеть с одним входом и одним выходом. Обучение методом градиентного спуска. Простейшая модель."""
############################################################

weight = 0.5
goal_pred = 0.8
input = 2
alpha = 0.1

for iteration in range(20):
    pred = input * weight
    error = (pred - goal_pred) ** 2
    direction_and_amount = (pred - goal_pred) * input
    weight -= (direction_and_amount * alpha)

    print(f"{iteration + 1}) Prediction: {pred}    Error: {error}")

