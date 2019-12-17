import pickle


def read_txt(path):
    return open(path, 'r', encoding='utf-8').readlines()


def read_pckl(path):
    return pickle.load(open(path, "rb"))


def save_pckl(path, model) -> None:
    with open(path, 'wb') as file:
        pickle.dump(model, file)
