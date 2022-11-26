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


def main():
    filename = "data.csv"
    df = load_csv(filename)

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
    beta_coefs = Sk * (K1_m - K2_m)
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

    # Дослідження коефіцієнтів отриманої дискримінантної функції
    label = "Коефіцієнти дискримінантної функції"
    K1_res = df_K1_without_class_attr * beta_coefs
    K2_res = df_K2_without_class_attr * beta_coefs

    K1_res_m = K1_res.mean()
    K2_res_m = K2_res.mean()

    thresholds_values = (K1_res_m + K2_res_m) / 2

    general_set_results = df_without_class_attr * beta_coefs
    general_set_comparison_results = general_set_results >= thresholds_values

    incorrect_classified_instances = (general_set_comparison_results['X9'].astype("int") - df['Class']) != 0

    col_names = ["Ознака", "Пороговий коефіцієнт"]
    table_data = np.vstack((df_without_class_attr.columns, thresholds_values)).T

    print()
    print_task(3.9)
    print(label)
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid"))

    print("Число неправильно класифікованих примірників:", incorrect_classified_instances.sum())


if __name__ == '__main__':
    main()
