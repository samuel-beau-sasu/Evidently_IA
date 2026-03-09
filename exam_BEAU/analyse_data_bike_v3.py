
import datetime
import pandas as pd
import requests
import zipfile
import io

from sklearn import ensemble, model_selection
from dataclasses import dataclass, field
from typing import List

from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    RegressionPreset,
    TargetDriftPreset
    )
from evidently.ui.workspace import Workspace

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


@dataclass
class DataConfig:
    """Configuration et données pour le monitoring Evidently."""

    # Colonnes cibles
    target: str
    prediction: str

    # Features
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)

    # Datasets
    reference_jan11: pd.DataFrame = field(default_factory=pd.DataFrame)
    current_feb11: pd.DataFrame = field(default_factory=pd.DataFrame)


"""
 Etape 1 Script pour l'ingestion de données
"""


# custom functions
def _fetch_data() -> pd.DataFrame:
    content = requests.get(
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        verify=False,
    ).content
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(
            arc.open("hour.csv"), header=0, sep=",", parse_dates=["dteday"]
        )
    return raw_data


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(
            row.dteday.date(),
            datetime.time(row.hr)
            ),
        axis=1,
    )
    return raw_data


# load data
def load_data(ref_int, curr_int) -> DataConfig:
    """
    Loads and processes bike sharing data.

    Returns:
        DataConfig: dataclass containing datasets and feature configuration.
    """
    raw_data = _process_data(_fetch_data())

    # Découpage reference / current
    reference_jan11 = raw_data.loc[ref_int]
    current_feb11 = raw_data.loc[curr_int]

    return DataConfig(
        target="cnt",
        prediction="prediction",
        numerical_features=[
            "temp",
            "atemp",
            "hum",
            "windspeed",
            "mnth",
            "hr",
            "weekday",
        ],
        categorical_features=["season", "holiday", "workingday"],
        reference_jan11=reference_jan11,
        current_feb11=current_feb11,
    )


"""
Initialisation de colonne mapping
"""
def init_colomn_mapping():

    # config = load_data()
    reference_interval = slice("2011-01-01 00:00:00", "2011-01-28 23:00:00")
    current_interval = slice("2011-01-29 00:00:00", "2011-02-28 23:00:00")

    config = load_data(reference_interval, current_interval)

    # Initialize the column mapping object which evidently uses to know how the data is structured
    column_mapping = ColumnMapping()

    # Map the actual target and prediction column names in the dataset for evidently
    column_mapping.target = config.target
    column_mapping.prediction = config.prediction

    # Specify which features are numerical and which are categorical for the evidently report
    column_mapping.numerical_features = config.numerical_features
    column_mapping.categorical_features = config.categorical_features

    return column_mapping


"""
Entraînez un RandomForestRegressor sur les données d'entraînement
et de test des données de janvier 2011
"""
def train_model(regressor):

    # config = load_data()
    reference_interval = slice("2011-01-01 00:00:00", "2011-01-28 23:00:00")
    current_interval = slice("2011-01-29 00:00:00", "2011-02-28 23:00:00")

    config = load_data(reference_interval, current_interval)

    # Train test split ONLY on reference_jan11
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        config.reference_jan11[config.numerical_features + config.categorical_features],
        config.reference_jan11[config.target],
        test_size=0.3,
    )

    # Model training on reference_jan11
    regressor.fit(X_train, y_train)

    # Predictions train / test
    preds_train = regressor.predict(X_train)
    preds_test = regressor.predict(X_test)

    # référence
    X_train[config.target] = y_train
    X_train[config.prediction] = preds_train

    # données actuel
    X_test[config.target] = y_test
    X_test[config.prediction] = preds_test

    return (X_train, X_test)


"""
Etape 2 :  script qui va créer un rapport basé sur RegressionPreset
en considérant les données d'entraînement comme la référence
et les données de test comme l'ensemble de données actuel.
"""
def regression_model_report(
    reference_data, current_data, column_mapping):
    """
    Generates a regression performance report using Evidently.
    """
    # Create a report instance for regression with RegressionPreset()
    regression_performance_report = Report(metrics=[RegressionPreset()])

    # Run the report
    regression_performance_report.run(
        reference_data=reference_data.sort_index(),
        current_data=current_data.sort_index(),
        column_mapping=column_mapping,
    )

    # 3. Maintenant on peut extraire les résultats
    result = regression_performance_report.as_dict()

    # Après avoir lancé votre rapport habituel
    result = regression_performance_report.as_dict()
    r2_current = result["metrics"][0]["result"]["current"]["r2_score"]
    r2_ref = result["metrics"][0]["result"]["reference"]["r2_score"]

    print(f"✅ R2 Reference (Train): {r2_ref:.4f}")
    print(f"🚀 R2 Current (Test): {r2_current:.4f}")

    return regression_performance_report

'''
Scripts pour l'analyse de la dérive
'''
def drift_model_analysis(regressor, column_mapping, config):

    # 2. Entraînement du modèle de production (Baseline Janvier)
    features = config.numerical_features + config.categorical_features
    regressor.fit(
        config.reference_jan11[features], config.reference_jan11[config.target]
    )

    # 3. Prédictions pour Janvier (Reference)
    config.reference_jan11["prediction"] = regressor.predict(
        config.reference_jan11[features]
    )

    # 4. Prédictions pour Février (Current)
    # On isole la période cible de février
    current_period = config.current_feb11.copy()
    current_period["prediction"] = regressor.predict(current_period[features])

    # CRUCIAL : On retire l'index temporel pour éviter le bug de fréquence 'H'
    reference_drift = config.reference_jan11.reset_index(drop=True)

    # 5. Rapport de Performance (La baseline de Janvier)
    # Create a report instance for regression with RegressionPreset()
    performance_report = Report(metrics=[RegressionPreset()])

    performance_report.run(
        reference_data=None,  # reference_drift,
        current_data=reference_drift,  # current_drift,
        column_mapping=column_mapping,
    )

    # On retourne les deux objets rapports
    return performance_report


def drift_model_analysis_week(regressor, column_mapping, config):

    # 2. Entraînement du modèle de production (Baseline Janvier)
    features = config.numerical_features + config.categorical_features
    regressor.fit(
        config.reference_jan11[features], config.reference_jan11[config.target]
    )

    # 3. Prédictions pour Janvier (Reference)
    config.reference_jan11["prediction"] = regressor.predict(
        config.reference_jan11[features]
    )

    # 4. Prédictions pour Février (Current)
    # On isole la période cible de février
    current_period = config.current_feb11.copy()
    current_period["prediction"] = regressor.predict(current_period[features])

    # CRUCIAL : On retire l'index temporel pour éviter le bug de fréquence 'H'
    reference_drift = config.reference_jan11.reset_index(drop=True)
    current_drift = current_period.reset_index(drop=True)

    # 5. Rapport de Performance (La baseline de Janvier)
    # Create a report instance for regression with RegressionPreset()
    performance_report = Report(metrics=[RegressionPreset()])

    performance_report.run(
        reference_data=reference_drift,
        current_data=current_drift,
        column_mapping=column_mapping,
    )

    # On retourne les deux objets rapports
    return performance_report


def drift_taget_analysis_week(regressor, column_mapping, config):

    # 2. Entraînement du modèle de production (Baseline Janvier)
    features = config.numerical_features + config.categorical_features
    regressor.fit(
        config.reference_jan11[features], config.reference_jan11[config.target]
    )

    # 3. Prédictions pour Janvier (Reference)
    config.reference_jan11["prediction"] = regressor.predict(
        config.reference_jan11[features]
    )

    # 4. Prédictions pour Février (Current)
    # On isole la période cible de février
    current_period = config.current_feb11.copy()
    current_period["prediction"] = regressor.predict(current_period[features])

    # CRUCIAL : On retire l'index temporel pour éviter le bug de fréquence 'H'
    reference_drift = config.reference_jan11.reset_index(drop=True)
    current_drift = current_period.reset_index(drop=True)

    # 5. Rapport de Performance (La baseline de Janvier)
    # Create a report instance for regression with RegressionPreset()
    performance_report = Report(metrics=[TargetDriftPreset()])

    performance_report.run(
        reference_data=reference_drift,
        current_data=current_drift,
        column_mapping=column_mapping,
    )

    # On retourne les deux objets rapports
    return performance_report


def drift_data_analysis(regressor, column_mapping, config):

    # 2. Entraînement du modèle de production (Baseline Janvier)
    features = config.numerical_features + config.categorical_features
    regressor.fit(
        config.reference_jan11[features], config.reference_jan11[config.target]
    )

    # 3. Prédictions pour Janvier (Reference)
    config.reference_jan11["prediction"] = regressor.predict(
        config.reference_jan11[features]
    )

    # 4. Prédictions pour Février (Current)
    # On isole la période cible de février

    current_period = config.current_feb11.copy()
    current_period["prediction"] = regressor.predict(current_period[features])

    # CRUCIAL : On retire l'index temporel pour éviter le bug de fréquence 'H'
    reference_drift = config.reference_jan11.reset_index(drop=True)
    current_drift = current_period.reset_index(drop=True)

    # 6. Rapport de Dérive des données (Janvier vs Février)
    drift_report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )
    drift_report.run(
        reference_data=reference_drift,
        current_data=current_drift,
        column_mapping=column_mapping,
    )

    # On retourne les deux objets rapports
    return drift_report


def add_report_to_workspace(workspace,
                            project_name,
                            project_description,
                            report):
    """
    Adds a snapshot to an existing or new project in a workspace.
    """
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    # ✅ add_run accepte un Snapshot nouvelle API
    # workspace.add_run(project.id, report)
    workspace.add_report(project.id, report)  # pas add_run !
    print(f"New report added to project {project_name}")


if __name__ == "__main__":
    WORKSPACE_NAME = "exam-workspace"
    PROJECT_NAME = "regression_monitoring"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    '''
    Prepare reference and current datasets by taking a sample of 50 rows
    '''

    # Model training on reference_jan11
    regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)

    column_mapping = init_colomn_mapping()

    X_train, X_test = train_model(regressor)

    # Generate the regression performance report
    regression_report = regression_model_report(
        X_train, X_test, column_mapping
    )

    # save HTML
    regression_report.save_html("Model performance report.html")

    # Generate Drift report
    reference_interval = slice("2011-01-01 00:00:00", "2011-01-28 23:00:00")
    current_interval = slice("2011-01-29 00:00:00", "2011-02-28 23:00:00")
    config = load_data(reference_interval, current_interval)

    perf_report = drift_model_analysis(regressor,
                                       column_mapping,
                                       config)

    # save HTML
    perf_report.save_html("Model reference report.html")

    # Generate Drift report week 1
    reference_interval = slice("2011-01-01 00:00:00", "2011-01-28 23:00:00")
    current_interval = slice("2011-01-29 00:00:00", "2011-02-07 23:00:00")
    config = load_data(reference_interval, current_interval)

    perf_report_w1 = drift_model_analysis_week(regressor,
                                               column_mapping,
                                               config)

    # save HTML
    perf_report_w1.save_html("Model drift week 1 report.html")

    # Generate Drift report week 2
    reference_interval = slice("2011-01-01 00:00:00", "2011-01-28 23:00:00")
    current_interval = slice("2011-02-07 00:00:00", "2011-02-14 23:00:00")
    config = load_data(reference_interval, current_interval)

    perf_report_w2 = drift_model_analysis_week(regressor,
                                               column_mapping,
                                               config)

    # save HTML
    perf_report_w2.save_html("Model drift week 2 report.html")

    # Generate Drift report week 3
    reference_interval = slice("2011-01-01 00:00:00", "2011-01-28 23:00:00")
    current_interval = slice("2011-02-15 00:00:00", "2011-02-21 23:00:00")
    config = load_data(reference_interval, current_interval)

    perf_report_w3 = drift_model_analysis_week(regressor,
                                               column_mapping,
                                               config)

    # save HTML
    perf_report_w3.save_html("Model drift week 3 report.html")

    # Generate Drift target week 3
    reference_interval = slice("2011-01-01 00:00:00", "2011-01-28 23:00:00")
    current_interval = slice("2011-02-15 00:00:00", "2011-02-21 23:00:00")
    config = load_data(reference_interval, current_interval)

    perf_report_target_w3 = drift_taget_analysis_week(regressor,
                                                      column_mapping,
                                                      config)

    # save HTML
    perf_report_target_w3.save_html("Target drift week 3 report.html")

    # Generate Drift target week 1
    reference_interval = slice("2011-01-01 00:00:00", "2011-01-28 23:00:00")
    current_interval = slice("2011-01-29 00:00:00", "2011-02-07 23:00:00")
    config = load_data(reference_interval, current_interval)

    perf_report_target_w1 = drift_taget_analysis_week(
        regressor,
        column_mapping,
        config)

    # save HTML
    perf_report_target_w1.save_html("Target drift week 1 report.html")

    # Generate Drift Data report
    reference_interval = slice("2011-01-01 00:00:00", "2011-01-28 23:00:00")
    current_interval = slice("2011-02-15 00:00:00", "2011-02-21 23:00:00")
    config = load_data(reference_interval, current_interval)

    drift_report = drift_data_analysis(regressor, column_mapping, config)

    # save HTML
    drift_report.save_html("Data drift report.html")

    # Create and Add report to workspace
    workspace = Workspace.create(WORKSPACE_NAME)

    # Ajout du premier rapport (Performance)
    add_report_to_workspace(
        workspace,
        PROJECT_NAME,
        PROJECT_DESCRIPTION,
        regression_report
    )

    # Ajout du deuxième rapport (Drift)
    add_report_to_workspace(
        workspace,
        PROJECT_NAME,
        "Rapport Performance Janvier",
        perf_report
    )

    # Ajout du  rapport (Drift week 1, week 2, week 3)
    add_report_to_workspace(
        workspace,
        PROJECT_NAME,
        "Rapport Performance Fevrier S1",
        perf_report_w1
    )
    add_report_to_workspace(
        workspace,
        PROJECT_NAME,
        "Rapport Performance Fevrier S2",
        perf_report_w2
    )
    add_report_to_workspace(
        workspace,
        PROJECT_NAME,
        "Rapport Performance Fevrier S3",
        perf_report_w3
    )

    # Ajout du  rapport target Drift ( week 1, week 3)
    add_report_to_workspace(
        workspace,
        PROJECT_NAME,
        "Rapport Performance des cibles Fevrier S1",
        perf_report_target_w1,
    )
    add_report_to_workspace(
        workspace,
        PROJECT_NAME,
        "Rapport Performance des cibles Fevrier S3",
        perf_report_target_w3,
    )

    # Ajout du  rapport (Drift Data)
    add_report_to_workspace(
        workspace,
        PROJECT_NAME,
        "Rapport Drift Février vs Janvier",
        drift_report
    )
