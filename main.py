import pandas as pd
import numpy as np
from tabulate import tabulate
from itertools import combinations


def load_csv(filename):
    return pd.read_csv(filename)


def print_task(no):
    print("Завдання {}:".format(no))


def get_K1(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Class"] == 1]


def get_K2(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Class"] != 1]


def get_M_x_K(x_subset: pd.DataFrame) -> int:
    n = len(x_subset)
    return (1 / n) * x_subset.sum()


def get_M_x_K_for_all_x(df: pd.DataFrame) -> dict:
    dictionary = {}
    for header_label in df:
        dictionary[header_label] = get_M_x_K(df[header_label])
    return dictionary


def get_D_x_K(x_subset: pd.DataFrame, M_x_K: int) -> int:
    n = len(x_subset) - 1
    return (1 / n) * sum([*map(lambda x: (x - M_x_K) ** 2, x_subset.to_numpy())])


def get_D_x_K_for_all_x(df: pd.DataFrame, m_x_k: dict) -> dict:
    dictionary = {}
    for header_label in df:
        dictionary[header_label] = get_D_x_K(df[header_label], m_x_k[header_label])
    return dictionary


def get_r_xi_xl_K(m_x_k: dict) -> dict:
    dictionary = {}
    for comb_tuple in combinations(m_x_k, 2):
        value = m_x_k[comb_tuple[0]] / m_x_k[comb_tuple[1]]
        key = "{} = r * {}".format(comb_tuple[0], comb_tuple[1])
        dictionary[key] = value
    return dictionary


def get_M_G_K(m_x_K: np.array, beta_coefs: np.array) -> int:
    return sum(m_x_K * beta_coefs)


def get_D_G_K(d_x_K: np.array, beta_coefs: np.array) -> int:
    return sum(d_x_K * beta_coefs ** 2)


def classify(df: pd.DataFrame, beta_coefs: np.array, P_threshold: int) -> list:
    res_set = []
    df_without_class_attr = df.drop('Class', axis=1)
    for i in df.index:
        disc_fun_result = round(sum((df_without_class_attr.loc[i] * beta_coefs)), 2)
        real_class = 'K1' if df.loc[i]['Class'] == 1 else 'K2'
        predicted_class = 'K1' if disc_fun_result >= P_threshold else 'K2'
        res_set.append([disc_fun_result, real_class, predicted_class])
    return res_set


def print_stats(classification_res: list) -> None:
    real_K1 = [*filter(lambda res_set: res_set[1] == 'K1', classification_res)]
    real_K2 = [*filter(lambda res_set: res_set[1] == 'K2', classification_res)]
    real_K1_n = len(real_K1)
    real_K2_n = len(real_K2)
    print("Справжні класи: K1 = {}, K2 = {}". format(real_K1_n, real_K2_n))

    predicted_K1 = [*filter(lambda res_set: res_set[2] == 'K1', classification_res)]
    predicted_K2 = [*filter(lambda res_set: res_set[2] == 'K2', classification_res)]
    predicted_K1_n = len(predicted_K1)
    predicted_K2_n = len(predicted_K2)
    print("Прогнозовані класи: K1 = {}, K2 = {}".format(predicted_K1_n, predicted_K2_n))

    misclassified = [*filter(lambda res_set: res_set[2] != res_set[1], classification_res)]
    misclassified_n = len(misclassified)
    print("Кількість помилково класифікованих примірників = {}".format(misclassified_n))

    rel_error = misclassified_n / len(classification_res)
    print("Відносна помилка = {}".format(rel_error))

def main():
    training_set_filename = "training_set.csv"
    test_set_filename = "test_set.csv"
    df = load_csv(training_set_filename)
    test_df = load_csv(test_set_filename)

    # Classes dataframes views
    df_K1 = get_K1(df)
    df_K2 = get_K2(df)

    df_without_class_attr = df.drop('Class', axis=1)
    df_K1_without_class_attr = df_K1.drop('Class', axis=1)
    df_K2_without_class_attr = df_K2.drop('Class', axis=1)

    # M*[xi/K1], D*[xi/K1]
    label = "Математичне сподівання і дисперсія кожної ознаки для К1"
    m_x_K1 = get_M_x_K_for_all_x(df_K1_without_class_attr)
    d_x_K1 = get_D_x_K_for_all_x(df_K1_without_class_attr, m_x_K1)

    col_names = ["Ознака", "M*[xi/K1]", "D*[xi/K1]"]
    table_data = np.vstack((df_K1_without_class_attr.columns.to_numpy(), [*m_x_K1.values()], [*d_x_K1.values()])).T

    print()
    print_task(3.3)
    print(label)
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    # M*[xi/K2], D*[xi/K2]
    label = "Математичне сподівання і дисперсія кожної ознаки для К2"
    m_x_K2 = get_M_x_K_for_all_x(df_K2_without_class_attr)
    d_x_K2 = get_D_x_K_for_all_x(df_K2_without_class_attr, m_x_K2)

    col_names = ["Ознака", "M*[xi/K2]", "D*[xi/K2]"]
    table_data = np.vstack((df_K2_without_class_attr.columns.to_numpy(), [*m_x_K2.values()], [*d_x_K2.values()])).T

    print()
    print_task(3.4)
    print(label)
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    # r*[xi,xl/K1], r*[xi,xl/K2]
    label = "Значення коефіцієнтів парної кореляції між ознаками за умови, що екземпляр належить до К1"
    r_xi_xl_K1 = get_r_xi_xl_K(m_x_K1)
    r_xi_xl_K2 = get_r_xi_xl_K(m_x_K2)

    col_names = ["Ознаки", "Значення кореляції (r) для К1", "Значення кореляції (r) для К2"]
    table_data = np.vstack(([*r_xi_xl_K1.keys()], [*r_xi_xl_K1.values()], [*r_xi_xl_K2.values()])).T

    print()
    print_task(3.5)
    print(label)
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    # 3.6
    # (X3 = r * X4), (X1 = r * X9) - статистично незначимі

    # отримуємо бета коефіцієнти
    K1_m = df_K1_without_class_attr.mean()
    K2_m = df_K2_without_class_attr.mean()

    Sk1 = df_K1_without_class_attr.cov()
    Sk2 = df_K2_without_class_attr.cov()

    n1 = len(df_K1)
    n2 = len(df_K2)
    Sk = 1 / (n1 + n2 - 2) * (n1 * Sk1 + n2 * Sk2)

    inv_Sk = pd.DataFrame(np.linalg.inv(Sk.values), Sk.columns, Sk.index)
    beta_coefs = inv_Sk * (K1_m - K2_m)
    beta_coefs = np.diagonal(beta_coefs)

    # M*[G/K1], M*[G/K2]
    label = "Оцінки умовних математичних сподівань дискримінантної функції за умови, що екземпляр належить К1 або К2"
    m_g_K1 = get_M_G_K([*m_x_K1.values()], beta_coefs)
    m_g_K2 = get_M_G_K([*m_x_K2.values()], beta_coefs)

    print()
    print_task("3.7")
    print(label)
    print("M*[G/K1] =", m_g_K1)
    print("M*[G/K2] =", m_g_K2)

    # D*[G/K1], D*[G/K2]
    label = "Оцінки умовних дисперсій дискримінантної функції за умови, що екземпляр належить К1 або К2"
    d_g_K1 = get_D_G_K([*d_x_K1.values()], beta_coefs)
    d_g_K2 = get_D_G_K([*d_x_K2.values()], beta_coefs)

    print()
    print_task("3.8")
    print(label)
    print("D*[G/K1] =", d_g_K1)
    print("D*[G/K2] =", d_g_K2)

    # Дослідження оцінок коефіцієнтів отриманої дискримінантної функції
    label = "Оцінки коефіцієнтів дискримінантної функції"

    col_names = ["Ознака", "Оцінка коефіцієнту"]
    table_data = np.vstack((df_without_class_attr.columns, beta_coefs)).T

    print()
    print_task(3.9)
    print(label)
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    # 3.10 - 3.18
    P_threshold = 68

    classification_res_for_training_set = classify(df, beta_coefs, P_threshold)
    classification_res_for_test_set = classify(test_df, beta_coefs, P_threshold)

    col_names = ['Значення дискр. функії', 'Фактичний клас', 'Прогнозований клас']
    table_data = classification_res_for_training_set

    print()
    print_task('3.10 - 3.18')
    print()
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    table_data = classification_res_for_test_set
    print()
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    print()
    print("Обране значення порогу = {}".format(P_threshold))

    print()
    print("Класифікація навчальної вибірки")
    print()
    print_stats(classification_res_for_training_set)

    print()
    print("Класифікація тестової вибірки")
    print()
    print_stats(classification_res_for_test_set)


if __name__ == '__main__':
    main()
