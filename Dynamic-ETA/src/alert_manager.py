import sqlite3
from pathlib import Path


# Store the SQLite database outside the source-code folder.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_PATH = DATA_DIR / "alerts.db"


def get_connection():
    """Create a connection to the alerts database."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the alerts table if it does not already exist."""

    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            train_number TEXT NOT NULL,
            journey_date TEXT NOT NULL,

            station_code TEXT NOT NULL,
            station_name TEXT NOT NULL,

            phone_number TEXT NOT NULL,

            alert_type TEXT NOT NULL DEFAULT 'boarding',

            alert_before_minutes INTEGER NOT NULL DEFAULT 15,

            alert_sent INTEGER NOT NULL DEFAULT 0,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()
    connection.close()


def create_subscription(
    train_number: str,
    journey_date: str,
    station_code: str,
    station_name: str,
    phone_number: str,
    alert_type: str = "boarding",
    alert_before_minutes: int = 15,
):
    """
    Create a passenger's station-alert subscription.

    Returns:
        Dictionary containing the subscription.
    """

    initialize_database()

    connection = get_connection()

    # Check whether the same subscription already exists.
    existing = connection.execute(
        """
        SELECT *
        FROM alert_subscriptions
        WHERE train_number = ?
          AND journey_date = ?
          AND station_code = ?
          AND phone_number = ?
        """,
        (
            train_number,
            journey_date,
            station_code,
            phone_number,
        ),
    ).fetchone()

    if existing:
        connection.close()

        return {
            "success": False,
            "message": "This alert subscription already exists.",
            "subscription_id": existing["id"],
        }

    cursor = connection.execute(
        """
        INSERT INTO alert_subscriptions (
            train_number,
            journey_date,
            station_code,
            station_name,
            phone_number,
            alert_type,
            alert_before_minutes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            train_number,
            journey_date,
            station_code,
            station_name,
            phone_number,
            alert_type,
            alert_before_minutes,
        ),
    )

    connection.commit()

    subscription_id = cursor.lastrowid

    connection.close()

    return {
        "success": True,
        "message": "Station alert subscription created.",
        "subscription_id": subscription_id,
    }


def get_active_subscriptions(
    train_number: str,
    journey_date: str,
):
    """
    Get subscriptions that have not received their alert yet.
    """

    initialize_database()

    connection = get_connection()

    rows = connection.execute(
        """
        SELECT *
        FROM alert_subscriptions
        WHERE train_number = ?
          AND journey_date = ?
          AND alert_sent = 0
        """,
        (
            train_number,
            journey_date,
        ),
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def mark_alert_sent(subscription_id: int):
    """Mark a subscription's SMS alert as sent."""

    initialize_database()

    connection = get_connection()

    connection.execute(
        """
        UPDATE alert_subscriptions
        SET alert_sent = 1
        WHERE id = ?
        """,
        (subscription_id,),
    )

    connection.commit()
    connection.close()


def get_all_subscriptions():
    """Return all subscriptions. Useful for testing."""

    initialize_database()

    connection = get_connection()

    rows = connection.execute(
        """
        SELECT *
        FROM alert_subscriptions
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]