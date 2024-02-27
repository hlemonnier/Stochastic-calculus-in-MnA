import wrds
import pandas as pd
import numpy as np
import os
import time
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Définir le chemin du fichier Excel pour stocker les données
fichier_excel = 'donnees_activision.xlsx'

# Télécharger les données si le fichier n'existe pas
if not os.path.exists(fichier_excel):
    db = wrds.Connection(wrds_username='hugolmn')
    data = db.raw_sql(f"""
                      select date, prc as close
                      from crsp.dsf
                      where permco in (
                          select permco
                          from crsp.dsenames
                          where ticker = 'ATVI'
                          and date between '2021-01-01' and '2023-12-31'
                      )
                      and date between '2021-01-01' and '2023-12-31'
                      """)
    data.set_index('date', inplace=True)
    data.to_excel(fichier_excel)
else:
    # Charger les données depuis le fichier Excel
    data = pd.read_excel(fichier_excel, index_col='date')

# Nettoyer les données
data['returns'] = 100 * data['close'].pct_change().dropna()
data = data[~data['returns'].isnull() & ~np.isinf(data['returns'])]
data['realized_volatility'] = data['returns'].rolling(window=252).std() * np.sqrt(252)
# Définir les plages des paramètres pour la validation croisée
p_range = range(1, 4)
q_range = range(1, 4)
results = []

# Validation croisée pour trouver les meilleurs paramètres
for p in p_range:
    for q in q_range:
        split_idx = int(len(data) * 0.8)
        train, test = data['returns'].iloc[:split_idx], data['returns'].iloc[split_idx:]
        model = arch_model(train, mean='Zero', vol='EGARCH', p=p, q=q, dist='t')
        model_fit = model.fit(disp='off')
        forecast = model_fit.forecast(horizon=len(test), method='simulation', simulations=500)
        predictions = forecast.mean.iloc[-1].values
        mse = mean_squared_error(test, predictions[:len(test)])
        results.append((p, q, mse))

best_params = sorted(results, key=lambda x: x[2])[0]
print(f"Meilleurs paramètres (p, q) basés sur MSE: {best_params[0]}, {best_params[1]} avec MSE de {best_params[2]}")

# Ajuster le modèle final EGARCH(1,1) avec une distribution t de Student
final_model = arch_model(data['returns'], mean='Zero', vol='EGARCH', p=best_params[0], q=best_params[1], dist='t')
final_model_fit = final_model.fit(disp='off')
print(final_model_fit.summary())

# Calculer la volatilité quotidienne prévue par le modèle EGARCH pour l'ensemble des données
forecast = final_model_fit.forecast(start='2021-01-01', method='simulation', simulations=100000)
predicted_volatility = (np.sqrt(forecast.variance.dropna().mean(axis=1)))

# Tracer les rendements réels et la volatilité quotidienne prévue
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['returns'], label='Rendements Réels', alpha=0.75)
plt.plot(predicted_volatility.index, predicted_volatility, label='Volatilité Quotidienne Prévue', color='red',
         alpha=0.75)
plt.title('Rendements Réels et Volatilité Quotidienne Prévue par le Modèle EGARCH(1,1)')
plt.legend()
plt.show()

# Estimer la volatilité annualisée à partir de la volatilité quotidienne moyenne prévue
annualized_volatility = predicted_volatility.mean() * np.sqrt(252)
print(f"Volatilité annualisée estimée: {annualized_volatility}")

# Supposons que `predicted_volatility` est un numpy array ou une liste de prévisions de volatilité
# Convertissez `predicted_volatility` en une série Pandas avec le même index que `data`
predicted_volatility_series = pd.Series(predicted_volatility, index=data.index[-len(predicted_volatility):])

# Assurez-vous que `realized_volatility` n'a pas de valeurs NaN pour la période que vous comparez
realized_volatility_series = data['realized_volatility'].dropna()

# Trouvez l'intersection des indices pour s'assurer que les séries sont alignées
common_index = realized_volatility_series.index.intersection(predicted_volatility_series.index)
realized_volatility_aligned = realized_volatility_series.loc[common_index]
predicted_volatility_aligned = predicted_volatility_series.loc[common_index]

# Maintenant, vous pouvez calculer MSE, MAE, et QLIKE sur les séries alignées
mse = mean_squared_error(realized_volatility_aligned, predicted_volatility_aligned)
print(f"MSE: {mse}")

mae = mean_absolute_error(realized_volatility_aligned, predicted_volatility_aligned)
print(f"MAE: {mae}")


def qlike(realized, predicted):
    return np.mean((realized - predicted) ** 2 / predicted)


qlike_score = qlike(realized_volatility_aligned, predicted_volatility_aligned)
print(f"QLIKE: {qlike_score}")

# Le code réalisé peut être utilisé dans le cadre de M&A pour évaluer le risque lié à la volatilité des actifs des
# entreprises concernées. En appliquant un modèle EGARCH pour simuler la volatilité d'une entreprise spécifique comme
# Activision Blizzard, vous fournissez une analyse quantitative du risque financier avant, pendant et après une
# opération de fusion ou acquisition. Cette approche permet d'appréhender de manière plus précise les fluctuations
# potentielles de valeur des actifs ciblés, offrant ainsi une base solide pour la prise de décision stratégique dans
# les opérations de M&A.
