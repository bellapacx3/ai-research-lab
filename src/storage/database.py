
"""SQLite database initialization for the AI Research Lab."""

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("artifacts/research_lab.db")


def get_connection(db_path=DEFAULT_DB_PATH):
    """Open a SQLite connection and enable foreign keys."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_database(db_path=DEFAULT_DB_PATH):
    """Create or safely upgrade the research database."""
    with get_connection(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT NOT NULL,
                dataset_path TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL,
                random_seed INTEGER NOT NULL,
                parameters_json TEXT NOT NULL,
                environment_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
            );

            CREATE TABLE IF NOT EXISTS model_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                mae REAL NOT NULL,
                rmse REAL NOT NULL,
                r2 REAL,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(id)
                    ON DELETE CASCADE,
                UNIQUE (experiment_id, model_name)
            );

            CREATE TABLE IF NOT EXISTS cross_validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                mae_mean REAL NOT NULL,
                mae_std REAL,
                rmse_mean REAL NOT NULL,
                rmse_std REAL,
                r2_mean REAL,
                r2_std REAL,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(id)
                    ON DELETE CASCADE,
                UNIQUE (experiment_id, model_name)
            );

            CREATE TABLE IF NOT EXISTS cross_validation_folds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                fold INTEGER NOT NULL,
                train_rows INTEGER NOT NULL,
                validation_rows INTEGER NOT NULL,
                mae REAL NOT NULL,
                rmse REAL NOT NULL,
                r2 REAL,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(id)
                    ON DELETE CASCADE,
                UNIQUE (experiment_id, model_name, fold)
            );

            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                sample_index TEXT NOT NULL,
                actual_value REAL NOT NULL,
                predicted_value REAL NOT NULL,
                residual REAL NOT NULL,
                absolute_error REAL NOT NULL,
                squared_error REAL NOT NULL,

                FOREIGN KEY (experiment_id, model_name)
                    REFERENCES model_results(experiment_id, model_name)
                    ON DELETE CASCADE,

                UNIQUE (experiment_id, model_name, sample_index),

                CHECK (absolute_error >= 0),
                CHECK (squared_error >= 0)
            );

            CREATE INDEX IF NOT EXISTS idx_experiments_created_at
                ON experiments(created_at);

            CREATE INDEX IF NOT EXISTS idx_model_results_experiment_id
                ON model_results(experiment_id);

            CREATE INDEX IF NOT EXISTS idx_cv_results_experiment_id
                ON cross_validation_results(experiment_id);

            CREATE INDEX IF NOT EXISTS idx_cv_folds_experiment_id
                ON cross_validation_folds(experiment_id);

            CREATE INDEX IF NOT EXISTS idx_predictions_experiment_model
                ON model_predictions(experiment_id, model_name);
            """
        )

        # Backward-compatible migration for existing databases.
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(model_results)"
            ).fetchall()
        }

        if "checkpoint_path" not in columns:
            connection.execute(
                "ALTER TABLE model_results ADD COLUMN checkpoint_path TEXT"
            )
