import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from math import sqrt

# Wczytaj dane (dostosuj ścieżkę, jeśli plik ma inną nazwę)
try:
    bike_data = pd.read_csv('C:\\Users\mpiesio\Desktop\KODILLA\wizualizacja\daily-bike-share.csv')
except FileNotFoundError:
    print("Błąd: Plik bike_data.csv nie istnieje. Podaj poprawną ścieżkę.")
    exit()

# Definiuj zmienne (dostosuj do swojego zbioru danych)
target = 'rentals'  # Kolumna docelowa
numeric_features = ['temp']  # Cechy numeryczne (dostosuj, np. ['temp', 'humidity'])
categorical_features = []  # Cechy kategoryczne (dostosuj, np. ['weather'])

# Pierwszy model: Prosta regresja liniowa z 'temp'
X = bike_data[['temp']].copy()
y = bike_data[target].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_first_train = lr_model.predict(X_train)
y_pred_first_test = lr_model.predict(X_test)

# Końcowy model: ElasticNet z cechami numerycznymi i kategorycznymi
X = bike_data[numeric_features + categorical_features].copy()
y = bike_data[target].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Definiuj transformatory
numeric_transformer = Pipeline(steps=[
    ('logtransformer', PowerTransformer()),
    ('standardscaler', StandardScaler()),
    ('polynomialfeatures', PolynomialFeatures())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline dla ElasticNet
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet())
])

# Hiperparametry dla GridSearchCV
params = {
    'preprocessor__num__polynomialfeatures__degree': [1, 2, 3, 4, 5],
    'regressor__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
    'regressor__l1_ratio': np.arange(0, 1.1, 0.1)
}

# Walidacja krzyżowa
cv = KFold(n_splits=5, shuffle=False)

# GridSearchCV
final_polynomial_regression_gridsearch = GridSearchCV(
    final_pipeline,
    params,
    scoring='neg_mean_squared_error',
    cv=cv
)

# Trenuj model
final_polynomial_regression_gridsearch.fit(X_train, y_train)
print("\nNajlepsze hiperparametry:", final_polynomial_regression_gridsearch.best_params_, "\n")

# Pobierz najlepszy model
final_polynomial_regression_model = final_polynomial_regression_gridsearch.best_estimator_

# Predykcje
y_pred_final_train = final_polynomial_regression_model.predict(X_train)
y_pred_final_test = final_polynomial_regression_model.predict(X_test)

# Obliczenie metryk
# Pierwszy model
r2_first_train = r2_score(y_train, y_pred_first_train)
r2_first_test = r2_score(y_test, y_pred_first_test)
mae_first_train = mean_absolute_error(y_train, y_pred_first_train)
mae_first_test = mean_absolute_error(y_test, y_pred_first_test)
mape_first_train = mean_absolute_percentage_error(y_train, y_pred_first_train)
mape_first_test = mean_absolute_percentage_error(y_test, y_pred_first_test)
mse_first_train = mean_squared_error(y_train, y_pred_first_train)
mse_first_test = mean_squared_error(y_test, y_pred_first_test)
rmse_first_train = sqrt(mse_first_train)
rmse_first_test = sqrt(mse_first_test)

# Końcowy model
r2_final_train = r2_score(y_train, y_pred_final_train)
r2_final_test = r2_score(y_test, y_pred_final_test)
mae_final_train = mean_absolute_error(y_train, y_pred_final_train)
mae_final_test = mean_absolute_error(y_test, y_pred_final_test)
mape_final_train = mean_absolute_percentage_error(y_train, y_pred_final_train)
mape_final_test = mean_absolute_percentage_error(y_test, y_pred_final_test)
mse_final_train = mean_squared_error(y_train, y_pred_final_train)
mse_final_test = mean_squared_error(y_test, y_pred_final_test)
rmse_final_train = sqrt(mse_final_train)
rmse_final_test = sqrt(mse_final_test)

# Wyświetlenie metryk
print("=== Pierwszy model (prosta regresja liniowa z 'temp') ===")
print(f"R² treningowe: {r2_first_train:.4f}")
print(f"R² testowe: {r2_first_test:.4f}")
print(f"MAE treningowe: {mae_first_train:.2f}")
print(f"MAE testowe: {mae_first_test:.2f}")
print(f"MAPE treningowe: {mape_first_train:.4f}")
print(f"MAPE testowe: {mape_first_test:.4f}")
print(f"MSE treningowe: {mse_first_train:.2f}")
print(f"MSE testowe: {mse_first_test:.2f}")
print(f"RMSE treningowe: {rmse_first_train:.2f}")
print(f"RMSE testowe: {rmse_first_test:.2f}")

print("\n=== Końcowy model (ElasticNet z cechami numerycznymi i kategorycznymi) ===")
print(f"R² treningowe: {r2_final_train:.4f}")
print(f"R² testowe: {r2_final_test:.4f}")
print(f"MAE treningowe: {mae_final_train:.2f}")
print(f"MAE testowe: {mae_final_test:.2f}")
print(f"MAPE treningowe: {mape_final_train:.4f}")
print(f"MAPE testowe: {mape_final_test:.4f}")
print(f"MSE treningowe: {mse_final_train:.2f}")
print(f"MSE testowe: {mse_final_test:.2f}")
print(f"RMSE treningowe: {rmse_final_train:.2f}")
print(f"RMSE testowe: {rmse_final_test:.2f}")

# Wizualizacje dla końcowego modelu
# 1. Wykres punktowy: przewidywane vs. rzeczywiste
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_final_test, color='blue', alpha=0.5, label='Predykcje')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideał')
plt.xlabel('Wartości rzeczywiste')
plt.ylabel('Wartości przewidywane')
plt.title('Końcowy model: Przewidywane vs. Rzeczywiste (Test)')
plt.legend()
plt.show()

# 2. Wykres reszt
errors_final = y_pred_final_test - y_test
plt.figure(figsize=(8, 6))
plt.scatter(y_test, errors_final, color='blue', alpha=0.25)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Wartości rzeczywiste')
plt.ylabel('Reszty')
plt.title('Końcowy model: Wykres reszt (Test)')
plt.show()

# 3. Histogram reszt
plt.figure(figsize=(8, 6))
plt.hist(errors_final, bins=20, color='blue', alpha=0.7)
plt.axvline(errors_final.mean(), color='black', linestyle='dashed', linewidth=1)
plt.xlabel('Reszty')
plt.ylabel('Częstość')
plt.title(f'Końcowy model: Histogram reszt (Średnia = {np.round(errors_final.mean(), 2)})')
plt.show()

