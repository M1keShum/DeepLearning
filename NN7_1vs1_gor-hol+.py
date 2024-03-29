"""Нейронная сеть с одним входом и одним выходом. Обучение методом горячо/холодно. Более сложное предсказание."""
############################################################

"""Чистая сеть"""
weight = 0.5  # Вес
lr = 0.001  # Коэффициент, на который изменится вес

number_of_toes = [0.5]  # Данные на вход
win_or_lose_binary = [0.8]  # Статистика о том, как произошло на самом деле, а не в предсказании

input = number_of_toes[0]  # Вход
true = win_or_lose_binary[0]  # На самом деле


def neural_network(input, weight):  # Нейронка, рассчитывающая предсказание
    prediction = input * weight
    return prediction


for i in range(1101):
    """Прогнозирование"""
    pred = neural_network(input, weight)  # Предсказание

    error = (pred - true) ** 2  # Вычисление ошибки. Метод вычисления среднеквадратической ошибки
    print(f"{i+1}) Error: {error}    Prediction: {pred}")

    """Сравнение"""
    p_up = neural_network(input, weight + lr)  # Предсказание при повышеннов весе
    e_up = (p_up - true) ** 2  # Ошибка при повышенном весе

    p_dn = neural_network(input, weight - lr)  # Предсказание при пониженном весе
    e_dn = (p_dn - true) ** 2  # Ошибка при пониженном весе

    """Сравнение + обучение. Сравнение ошибок и выбор нового значения веса"""
    if error > e_dn or error > e_up:  # Если хоть при 1 варианте ошибка меньше, то...
        if e_dn < e_up:  # Если при пониженном весе ошибка меньше, то...
            weight -= lr
        if e_up < e_dn:  # Если при повышенном весе ошибка меньше, то...
            weight += lr




















