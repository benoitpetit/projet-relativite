import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.compose import ColumnTransformer
from astropy import constants as const
from main import generate_dataset, create_preprocessor, main


def test_dataset_generation():
    """Vérifie l'intégrité structurelle des données générées"""
    data = generate_dataset(100, random_seed=42)

    assert data.shape == (100, 3), "Structure de données incorrecte"
    assert 'v_norm' in data.columns, "Colonne v_norm manquante"
    assert np.all(data['v_norm'].between(-0.99, 0.99,
                  inclusive='both')), "Vitesse non relativiste"
    assert np.all(data['mass'].between(1e-30, 1e-25)
                  ), "Masse hors échelle quantique"


def test_preprocessor_pipeline():
    """Teste la transformation complète des données"""
    preprocessor = create_preprocessor()
    test_data = pd.DataFrame({
        'v_norm': [0.0, 0.5, 0.99],
        'mass': [1e-30, 5e-27, 1e-25]
    })

    transformed = preprocessor.fit_transform(test_data)

    assert transformed.shape == (3, 2), "Dimensions de sortie incorrectes"
    assert np.allclose(transformed.mean(axis=0), 0,
                       atol=1e-7), "Centrage incorrect"
    assert np.allclose(transformed.std(axis=0), 1,
                       atol=1e-5), "Réduction d'échelle incorrecte"


@pytest.mark.parametrize("velocity", [-0.99, 0.0, 0.99])
def test_velocity_edge_cases(velocity):
    """Vérifie la génération des vitesses limites"""
    data = generate_dataset(1000, random_seed=42)
    assert np.any(np.isclose(data['v_norm'], velocity, atol=0.01)), f"Cas {
        velocity}c non couvert"


def test_relativistic_energy_calculation():
    """Validation de la formule relativiste au repos (v=0)"""
    c = const.c.value
    mass_test = 1e-27  # kg
    data = pd.DataFrame({
        'v_norm': [0.0],
        'mass': [mass_test],
        'energy': mass_test * c**2
    })

    assert np.isclose(data.energy[0], 8.987551787e-11,
                      rtol=1e-9), "Erreur énergie au repos"


def test_full_pipeline_integration(tmp_path):
    """Test d'intégration complète du pipeline avec vérification du format de base de données"""
    main()

    assert Path('analysis.json').exists(), "Fichier d'analyse manquant"

    with open('analysis.json') as f:
        results = json.load(f)
        assert 'analyses' in results, "Clé 'analyses' absente dans le fichier"
        assert len(results['analyses']) > 0, "Aucune analyse sauvegardée"
        first_analysis = results['analyses'][0]
        assert 'training_history' in first_analysis, "Historique d'entraînement absent"
        assert first_analysis['performance']['r2'] > 0.7, "Performance modèle insuffisante"
        assert first_analysis['training_metrics']['converged'], "Modèle non convergent"
