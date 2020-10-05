"""Нейронная сеть с одним входом и одним выходом"""
#####################################################

weight = 0.1  # Весовой коэффициент


def neural_network(input, weight):  # Сама нейронная сеть
    prediction = input * weight  # Рассчёт предсказания
    return prediction


number_of_toes = [8.5, 9.5, 10, 9]  # Набор данных

input = number_of_toes[0]  # Точка данных

pred = neural_network(input, weight)
print(pred)
