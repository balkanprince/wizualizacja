import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from numpy import where, meshgrid, arange, hstack

# Funkcja do wizualizacji granic decyzyjnych
def plot_classification_surface(X_plot, y_plot, trained_model):
    plt.figure(figsize=(12, 7))
    min1, max1 = X_plot[:, 0].min()-1, X_plot[:, 0].max()+1
    min2, max2 = X_plot[:, 1].min()-1, X_plot[:, 1].max()+1
    x1grid = arange(min1, max1, 0.1)
    x2grid = arange(min2, max2, 0.1)
    xx, yy = meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = hstack((r1, r2))
    yhat = trained_model.predict(grid)
    zz = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap='Paired')
    for class_value in range(2):
        row_ix = where(y_plot == class_value)
        plt.scatter(X_plot[row_ix, 0], X_plot[row_ix, 1], cmap='Paired', alpha=0.3, label=class_value)
    plt.legend(loc='upper right')
    plt.show()

# Wczytanie danych
data = pd.read_csv('diabetes.csv')

# Wyświetlenie nazw kolumn, aby sprawdzić strukturę danych
print("Nazwy kolumn w pliku diabetes.csv:")
print(data.columns)

# Zaktualizuj nazwę kolumny docelowej na podstawie wydruku powyżej
# Przykład: jeśli kolumna docelowa nazywa się 'Diabetic', użyj:
target_column = 'Diabetic'  # Zmień na poprawną nazwę kolumny
y = data[target_column]

# Wybór dwóch cech (sprawdź, czy te nazwy istnieją w Twoich danych)
feature_columns = ['Pregnancies', 'Age']  # Zmień, jeśli nazwy są inne
X = data[feature_columns]

# Podział danych na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)

# Standaryzacja danych
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

# Lista regularyzacji i wartości C
penalties = ['l1', 'l2', 'elasticnet', 'none']
C_values = [0.01, 1, 100]
l1_ratio = 0.5  # Dla elasticnet

# Przechowywanie wyników
results = []

# Pętla po różnych konfiguracjach
for penalty in penalties:
    for C in C_values:
        # Konfiguracja solvera i parametrów
        if penalty in ['l1', 'elasticnet']:
            solver = 'saga'  # Wymagany dla L1 i elasticnet
        else:
            solver = 'lbfgs'  # Domyślny dla L2 i none
        
        # Dla penalty='none' ignorujemy C
        if penalty == 'none':
            model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000)
            config_name = f"penalty=None"
        else:
            if penalty == 'elasticnet':
                model = LogisticRegression(penalty=penalty, C=C, solver=solver, l1_ratio=l1_ratio, max_iter=10000)
                config_name = f"penalty={penalty}, C={C}, l1_ratio={l1_ratio}"
            else:
                model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=10000)
                config_name = f"penalty={penalty}, C={C}"
        
        # Trenowanie modelu
        model.fit(X_train_standardized, y_train)
        
        # Obliczenie F1-score na zbiorze treningowym i testowym
        train_pred = model.predict(X_train_standardized)
        test_pred = model.predict(X_test_standardized)
        f1_train = f1_score(y_train, train_pred)
        f1_test = f1_score(y_test, test_pred)
        
        # Zapisywanie wyników
        results.append({
            'Config': config_name,
            'F1_Train': f1_train,
            'F1_Test': f1_test
        })
        
        # Wizualizacja granic decyzyjnych
        print(f"\nKonfiguracja: {config_name}")
        print(f"F1-score (trening): {f1_train:.4f}")
        print(f"F1-score (test): {f1_test:.4f}")
        plot_classification_surface(X_plot=X_train_standardized, y_plot=y_train, trained_model=model)

# Wyświetlenie wyników w formie tabeli
results_df = pd.DataFrame(results)
print("\nPodsumowanie wyników:")
print(results_df)