import json
import numpy as np
import pandas as pd
from astropy import constants as const
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def generate_dataset(num_samples=10000, random_seed=None):
    """Génère des données relativistes avec contrôle de la randomisation"""
    rng = np.random.default_rng(random_seed)
    c = const.c.value

    data = pd.DataFrame({
        'velocity': rng.uniform(-0.99*c, 0.99*c, num_samples),
        'mass': rng.uniform(1e-30, 1e-25, num_samples)
    })

    data['v_norm'] = data.velocity / c
    data['energy'] = data.mass * c**2 / np.sqrt(1 - data.v_norm**2)

    return data[['v_norm', 'mass', 'energy']]


def create_preprocessor():
    """Crée un pipeline de prétraitement avec conservation des métadonnées"""
    return ColumnTransformer([
        ('scaler', StandardScaler(), ['v_norm', 'mass'])
    ])


def save_analysis_data(model, X_test, y_test, cv_scores, target_scaler, filename='analysis.json'):
    """Sauvegarde en mode base de données contenant toutes les analyses lancées"""
    y_pred = model.predict(X_test)
    y_pred_orig = target_scaler.inverse_transform(
        y_pred.reshape(-1, 1)).flatten()
    y_test_orig = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)).flatten()

    new_entry = {
        'cross_validation': {
            'mean_r2': float(np.mean(cv_scores)),
            'std_r2': float(np.std(cv_scores)),
            'scores': [float(score) for score in cv_scores]
        },
        'model_architecture': {
            'hidden_layers': list(model.hidden_layer_sizes),
            'activation': model.activation
        },
        'performance': {
            'mse': float(mean_squared_error(y_test_orig, y_pred_orig)),
            'r2': float(r2_score(y_test_orig, y_pred_orig))
        },
        'training_history': model.loss_curve_,
        'training_metrics': {
            'final_loss': float(model.loss_),
            'converged': model.n_iter_ < model.max_iter
        }
    }

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'analyses' not in data:
            data = {'analyses': [data]}
    except FileNotFoundError:
        data = {'analyses': []}

    data['analyses'].append(new_entry)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Nouvelle analyse sauvegardée. Total actuel: {
          len(data['analyses'])} runs")


def main():
    """Pipeline principal avec boucle pour lancer plusieurs analyses"""
    n_runs = 5
    for run in range(n_runs):
        print(f"--- Exécution {run+1}/{n_runs} ---")
        # Utiliser un seed différent pour chaque run
        current_seed = 42 + run
        data = generate_dataset(random_seed=current_seed)
        X = data[['v_norm', 'mass']]
        y = data['energy'].values.reshape(-1, 1)

        # Normalisation des features et target
        preprocessor = create_preprocessor()
        target_scaler = StandardScaler()

        X_processed = preprocessor.fit_transform(X)
        y_processed = target_scaler.fit_transform(y).flatten()

        # Configuration modèle optimisée
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            early_stopping=True,
            max_iter=2000,
            random_state=current_seed  # Changer aussi le random_state du modèle
        )

        # Validation croisée
        cv_scores = cross_val_score(
            model, X_processed, y_processed, cv=5, scoring='r2')
        print(f"R² moyen par validation croisée : {
              np.mean(cv_scores):.2f} (±{np.std(cv_scores):.2f})")

        # Entraînement final
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed,
            test_size=0.2,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Sauvegarde avec dénormalisation
        save_analysis_data(model, X_test, y_test, cv_scores, target_scaler)
    print("Toutes les analyses ont été sauvegardées dans analysis.json")


if __name__ == "__main__":
    main()
