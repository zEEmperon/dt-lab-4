import pandas as pd


def load_csv(filename):
    return pd.read_csv(filename)


def print_task(no):
    print("Завдання {}:".format(no))


def main():
    filename = "data.csv"
    df = load_csv(filename)
    print(df)


if __name__ == '__main__':
    main()
