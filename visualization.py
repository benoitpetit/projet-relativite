# visualization.py
from main import generate_dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor  # Import manquant
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sns.set(style='whitegrid', context='notebook', palette='muted')


def load_analysis_data(filename='analysis.json'):
    """Charge les données d'analyse précédemment sauvegardées"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['analyses']


def plot_training_history(analyses, output_dir='visualizations'):
    """Visualise l'historique d'entraînement de tous les runs"""
    Path(output_dir).mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(analyses)))
    
    # Graphique principal
    for i, (run, color) in enumerate(zip(analyses, colors)):
        history = run['training_history']
        ax1.plot(history, label=f'Run {i+1}', color=color, alpha=0.8, linewidth=2)
        
        # Annotation de la perte finale
        final_loss = history[-1]
        ax1.annotate(f'Loss finale {i+1}: {final_loss:.6f}',
                    xy=(len(history)-1, final_loss),
                    xytext=(10, 20+i*20), 
                    textcoords='offset points',
                    color=color,
                    fontsize=8,
                    arrowprops=dict(arrowstyle='->',
                                  connectionstyle='arc3,rad=.2',
                                  color=color))

    ax1.set_title('Évolution de la perte pendant l\'entraînement\n(Vue complète)',
                 pad=20, fontsize=12)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Zoom sur la convergence finale
    for i, (run, color) in enumerate(zip(analyses, colors)):
        history = run['training_history']
        # Afficher seulement le dernier tiers des époques
        start_idx = len(history) // 3 * 2
        ax2.plot(range(start_idx, len(history)),
                history[start_idx:],
                label=f'Run {i+1}',
                color=color,
                alpha=0.8,
                linewidth=2)

    ax2.set_title('Zoom sur la convergence finale\n(Dernier tiers)',
                 pad=20, fontsize=12)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss (MSE)')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(Path(output_dir)/'training_history.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def create_comprehensive_report(analyses, output_dir='visualizations'):
    """Génère un rapport complet avec visualisations et métriques clés"""
    report = {
        'performance_metrics': [{
            'run': i + 1,
            'mse': run['performance']['mse'],
            'r2': run['performance']['r2']
        } for i, run in enumerate(analyses)],
        'convergence_analysis': [],
        'physical_relationships': {},
        'model_comparison': {}
    }

    # Analyse comparative des runs
    metrics = [
        ('performance', 'mse'),
        ('performance', 'r2'),
        ('training_metrics', 'final_loss')
    ]

    for section, metric in metrics:
        values = [run[section][metric] for run in analyses]
        report['model_comparison'][metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    # Analyse de convergence
    convergence_data = []
    for run in analyses:
        history = run['training_history']
        # Convert NumPy types to native Python types
        convergence_point = int(
            np.argmin(np.abs(np.gradient(history))))  # Convert to int
        total_epochs = len(history)
        convergence_data.append({
            'final_loss': float(history[-1]),  # Ensure float type
            'convergence_epoch': convergence_point,
            # Ensure float
            'convergence_rate': float(convergence_point / total_epochs)
        })
    report['convergence_analysis'] = convergence_data

    # Génération des visualisations
    plot_performance_distribution(analyses, output_dir)
    plot_error_analysis(analyses, output_dir)
    plot_physical_relationships(output_dir)

    # Sauvegarde du rapport
    with open(Path(output_dir)/'analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    return report


def plot_performance_distribution(analyses, output_dir):
    """Visualise la distribution des performances"""
    plt.figure(figsize=(12, 4))

    # R² scores
    plt.subplot(131)
    r2_scores = [run['performance']['r2'] for run in analyses]
    sns.violinplot(y=r2_scores)
    plt.title('Distribution des scores R²')

    # MSE
    plt.subplot(132)
    mse_values = [run['performance']['mse'] for run in analyses]
    sns.boxplot(y=mse_values)
    plt.title('Distribution du MSE')
    plt.yscale('log')

    # Temps de convergence
    plt.subplot(133)
    conv_epochs = [len(run['training_history']) for run in analyses]
    sns.histplot(conv_epochs, kde=True)
    plt.title('Distribution des epochs\npour convergence')

    plt.tight_layout()
    plt.savefig(Path(output_dir)/'performance_distribution.png', dpi=300)
    plt.close()


def plot_error_analysis(analyses, output_dir):
    """Analyse des erreurs de prédiction"""
    preprocessor = ColumnTransformer([
        ('scaler', StandardScaler(), ['v_norm', 'mass'])
    ])
    target_scaler = StandardScaler()

    data = generate_dataset(1000, random_seed=42)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=2000,
        random_state=42
    )

    X_processed = preprocessor.fit_transform(data[['v_norm', 'mass']])
    y_processed = target_scaler.fit_transform(
        data[['energy']].values.reshape(-1, 1))

    model.fit(X_processed, y_processed.ravel())

    predictions = target_scaler.inverse_transform(
        model.predict(X_processed).reshape(-1, 1)
    )

    plt.figure(figsize=(10, 6))
    residuals = predictions.flatten() - data['energy']
    sns.scatterplot(x=data['energy'], y=residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Analyse des résidus')
    plt.xlabel('Valeur réelle de l\'énergie')
    plt.ylabel('Erreur de prédiction')
    plt.savefig(Path(output_dir)/'residual_analysis.png', dpi=300)
    plt.close()


def plot_physical_relationships(output_dir):
    """Visualise les relations physiques sous-jacentes"""
    data = generate_dataset(1000, random_seed=42)

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3)

    ax0 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(x=data['v_norm'], y=data['energy'],
                    hue=data['mass'], palette='viridis', ax=ax0)
    ax0.set_title('Relation énergie-vitesse')
    ax0.set_xlabel('Vitesse normalisée (v/c)')
    ax0.set_ylabel('Énergie (J)')

    ax1 = fig.add_subplot(gs[0, 1])
    sns.histplot(np.log10(data['mass']), kde=True, ax=ax1)
    ax1.set_title('Distribution des masses')
    ax1.set_xlabel('log10(Masse) [kg]')

    ax2 = fig.add_subplot(gs[0, 2])
    gamma = 1/np.sqrt(1 - data['v_norm']**2)
    sns.lineplot(x=data['v_norm'], y=gamma, ax=ax2)
    ax2.set_title('Facteur Lorentz en fonction de la vitesse')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(Path(output_dir)/'physical_relationships.png', dpi=300)
    plt.close()


def main():
    """Point d'entrée principal pour la génération des visualisations"""
    analyses = load_analysis_data()
    plot_training_history(analyses)
    report = create_comprehensive_report(analyses)
    print("Analyse complète générée dans le dossier 'visualizations'")


if __name__ == '__main__':
    main()
