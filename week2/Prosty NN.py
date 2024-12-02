# UWAGA DO PROGRAMU
# ponieważ generujemy dane losowe, nie mają one żadnego prawdziwego odzwierciedlenia
# i schematów, przez co sieć NIE wyternuje się poprawnie.
# Żeby sieć wytrenowała się poprawie musielibyśmy pozyskać dane faktyczne i uczyć oraz walidować na nich

# To jest tylko przykład jak pisać i ternować sieci.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def generate_data():
    # Losowanie danych temperatury, zachmurzenia i opadów
    temp = np.random.randint(-7, 13, size=20000)
    opad = np.random.uniform(0, 50.001, size=20000)
    chmury = np.random.random(size=20000)

    print(f"{len(temp)}, {len(opad)}, {len(chmury)}")
    # Zapisz w postaci CSV bez kolumny indeksu
    pd.DataFrame({"Temp": temp, "Opad": opad, "Chmury": chmury}).to_csv("./dane.csv", index=False)


device = "cuda" if torch.cuda.is_available() else "cpu"


class Network(nn.Module):
    def __init__(self):
        super().__init__()  # Inicjalizujemy wszystkie funkcje/zmienne nn.Module
        self.input = nn.Linear(2, 2000)  # Warstwa wejściowa na opady i zachmurzenie
        self.hidden1 = nn.Linear(2000, 4000)  # Warstwa ukryta
        self.hidden2 = nn.Linear(4000, 2000)  # Warstwa ukryta
        self.output = nn.Linear(2000, 1)  # Warstwa wyjściowa przewidująca temperature

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.output(x))
        return x


def standaryzacja(dane: npt.NDArray):
    srednia: np.float64 = dane.mean()
    odchylenie: np.float64 = dane.std()
    return (dane - srednia) / odchylenie, (srednia, odchylenie)


def przywrocenie_standaryzacji(standaryzowane: npt.NDArray, srednia: np.float64, odchylenie: np.float64) -> npt.NDArray:
    return standaryzowane * odchylenie + srednia

if __name__ == "__main__":
    GENERATE = True  # Flaga czy generować nowy zestaw danych
    if GENERATE:  # Jeśli flaga włączona generuj nowy zestaw danych
        generate_data()

    # Wczytaj zestaw danych
    dane = pd.read_csv("./dane.csv")
    dane_numpy = np.array(dane) # Konwersja pd.Dataframe na np.NDArray
    # Standaryzacja
    stand_temp, temp_mean_std = standaryzacja(dane_numpy[:, 0]) # Temperatura
    stand_opad, opad_mean_std = standaryzacja(dane_numpy[:, 1]) # Opady
    stand_chmury, chmury_mean_std = standaryzacja(dane_numpy[:, 2]) # Zachmurzenie
    # Złączenie ustandaryzowanych kolumn do jednej macierzy
    dane_standaryzowane = np.column_stack([stand_temp, stand_opad, stand_chmury])
    # Konwersja np.NDArray na torch.Tensor o typie float32
    # (float64 byłby tzw. Double, którego sieć nie obsłuży bez jej dodatkowej obróbki)
    dane_tensor = torch.tensor(dane_standaryzowane, dtype=torch.float32)
    dataset = TensorDataset(dane_tensor)  # Tworzenie zestawu danych

    # Podział danych na treningowe i walidacyjne
    train = 0.8 * len(dataset)
    val = len(dataset) - train
    data_train, data_val = random_split(dataset, [int(train), int(val)])
    # Tworzenie ładowaczy danych dla sieci
    batch = 64 # Ile danych w 1 batchu
    train_loader = DataLoader(data_train, batch, shuffle=True)
    val_loader = DataLoader(data_val, batch, shuffle=False)

    # TRENOWANIE
    TRAIN = True
    if TRAIN:
        net = Network()
        net.to(device) # prezniesienie sieci na urządzenie wykonania (CPU lub GPU)
        criterion = nn.L1Loss() # Kryterium porównania przewidywań z prawdą (Średni błąd bezwzględny - MAE)
        optimizer = optim.Adam(net.parameters(), lr=0.001) # Optymizer modelu, Adam jest generalnym, najczęściej używanym
        for epoch in range(100):

            for batch_i, data in enumerate(train_loader, 0):
                # Ładujemy nasze wejścia dla sieci i prawdziwe wartości
                # U nas wejścia są w kolumnach 1 i 2 (opady, zachmurzenie)
                # Temperatura jest używana jako kryterium prawdy (kolumna 0)
                # Konstrukt x[:, 1:3] oznacza ładuj wartości wszystkich wierszy z pominięciem 1 kolumny
                data = data[0]  # Wybieramy tensor z konstruktu data
                # Rozdział danych na wejście i prawde z przeniesieniem
                inputs, true_temp = data[:, 1:3].to(device), data[:, 0].to(device)
                # Zerujemy gradient parametrów
                optimizer.zero_grad()
                # Oblicz output
                outputs = net(inputs)
                # Oblicz straty przewidywań modelu i wykonaj propagację wsteczną
                loss = criterion(outputs, true_temp)
                loss.backward()
                # Optymalizuj model
                optimizer.step()

                # Wyświetl co robi program co 10 batch
                if batch_i % 10 == 9:
                    print(f"[Epoch {epoch + 1} -> Batch {batch_i + 1}] -> Loss: {loss.item():.4f}")

        print("Training Finished")
        # Zapisujemy CAŁY model
        model_path = "./temperature-model.pth"
        torch.save(net, model_path)

    # EWALUACJA
    EVAL = True
    if EVAL:
        # Ładujemy model do nowej klasy
        net_eval = torch.load("./temperature-model.pth", weights_only=False)
        net_eval.eval()
        net_eval.to(device)  # Przenosimy model na urządzenie wykonania (CPU lub GPU)
        # Nie trenujemy więc nie potrzebujemy obliczać gradietów
        all_outputs = []
        all_real_temps = []
        with torch.no_grad():
            for data in val_loader:
                # Analogicznie do trenowania, ładujemy inputy i naszą prawdę
                data = data[0]
                inputs, true_temp = data[:, 1:3].to(device), data[:, 0].to(device)
                # Przepuszczamy nasze dane przez sieć
                outputs = net_eval(inputs)
                # Przywracamy ustandaryzowane dane temperatur do normy zapisanymi wcześniej odchyleniai i średnimi
                przewidywane = przywrocenie_standaryzacji(np.array(outputs.cpu()), *temp_mean_std)
                rzeczywiste = przywrocenie_standaryzacji(np.array(true_temp.cpu()), *temp_mean_std)
                all_outputs.extend(przewidywane.tolist())
                all_real_temps.extend(rzeczywiste.tolist())
        mae = mean_absolute_error(all_real_temps, all_outputs)
        mse = mean_squared_error(all_real_temps, all_outputs)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_real_temps, all_outputs)
        print(f"Model wytrenował się z następującymi statystykami:\n{mse = }\n{mae = }\n{rmse = }\n{r2 = }")
        print(f"20 pierwszych przewidywań: {all_outputs[:20]}")
        print(f"20 pierwszych prawd: {all_real_temps[:20]}")




# generate_data()
