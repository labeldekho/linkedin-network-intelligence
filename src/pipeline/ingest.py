"""
LinkedIn Data Export Ingestion

Loads and validates CSV files from LinkedIn data export.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConnectionRecord(BaseModel):
    """Raw connection record from Connections.csv."""
    first_name: str
    last_name: str
    email_address: Optional[str] = None
    company: Optional[str] = None
    position: Optional[str] = None
    connected_on: Optional[datetime] = None
    url: Optional[str] = None


class MessageRecord(BaseModel):
    """Raw message record from Messages.csv."""
    conversation_id: Optional[str] = None
    conversation_title: Optional[str] = None
    from_name: str = Field(alias="from")
    sender_profile_url: Optional[str] = None
    to_name: Optional[str] = Field(default=None, alias="to")
    date: datetime
    subject: Optional[str] = None
    content: str


class EndorsementRecord(BaseModel):
    """Raw endorsement record from Endorsement_*.csv files."""
    endorser_name: Optional[str] = None
    endorsee_name: Optional[str] = None
    skill: Optional[str] = None
    date: Optional[datetime] = None
    status: Optional[str] = None


class RecommendationRecord(BaseModel):
    """Raw recommendation record from Recommendations_*.csv files."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    text: Optional[str] = None
    date: Optional[datetime] = None
    status: Optional[str] = None


class LinkedInExport(BaseModel):
    """Container for all loaded LinkedIn export data."""
    connections: list[ConnectionRecord] = Field(default_factory=list)
    messages: list[MessageRecord] = Field(default_factory=list)
    endorsements_given: list[EndorsementRecord] = Field(default_factory=list)
    endorsements_received: list[EndorsementRecord] = Field(default_factory=list)
    recommendations_given: list[RecommendationRecord] = Field(default_factory=list)
    recommendations_received: list[RecommendationRecord] = Field(default_factory=list)

    # Metadata
    source_directory: Optional[str] = None
    loaded_files: list[str] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def has_connections(self) -> bool:
        return len(self.connections) > 0

    @property
    def has_messages(self) -> bool:
        return len(self.messages) > 0


def _parse_date(date_str: Optional[str], formats: list[str] = None) -> Optional[datetime]:
    """Parse date string with multiple format support."""
    if date_str is None or pd.isna(date_str):
        return None

    formats = formats or [
        "%d %b %Y",      # LinkedIn format: "15 Jan 2024"
        "%Y-%m-%d",      # ISO format
        "%m/%d/%Y",      # US format
        "%d/%m/%Y",      # EU format
        "%B %d, %Y",     # Full month: "January 15, 2024"
        "%Y-%m-%d %H:%M:%S",  # With time
    ]

    date_str = str(date_str).strip()

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    logger.warning(f"Could not parse date: {date_str}")
    return None


def _clean_name(name: Optional[str]) -> str:
    """Clean and normalize a name string."""
    if name is None or pd.isna(name):
        return ""
    return str(name).strip()


def _load_connections(filepath: Path) -> list[ConnectionRecord]:
    """Load Connections.csv file."""
    records = []

    try:
        # Skip first 3 rows (LinkedIn export header)
        df = pd.read_csv(filepath, skiprows=3)

        # Normalize column names
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        for _, row in df.iterrows():
            try:
                record = ConnectionRecord(
                    first_name=_clean_name(row.get("first_name")),
                    last_name=_clean_name(row.get("last_name")),
                    email_address=row.get("email_address") if pd.notna(row.get("email_address")) else None,
                    company=row.get("company") if pd.notna(row.get("company")) else None,
                    position=row.get("position") if pd.notna(row.get("position")) else None,
                    connected_on=_parse_date(row.get("connected_on")),
                    url=row.get("url") if pd.notna(row.get("url")) else None,
                )
                records.append(record)
            except Exception as e:
                logger.warning(f"Skipping malformed connection row: {e}")

        logger.info(f"Loaded {len(records)} connections from {filepath.name}")

    except Exception as e:
        logger.error(f"Error loading connections from {filepath}: {e}")
        raise

    return records


def _load_messages(filepath: Path) -> list[MessageRecord]:
    """Load Messages.csv file."""
    records = []

    try:
        df = pd.read_csv(filepath)

        # Normalize column names
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        for _, row in df.iterrows():
            try:
                # Handle different column name variations
                from_name = row.get("from") or row.get("from_name") or row.get("sender") or ""
                to_name = row.get("to") or row.get("to_name") or row.get("recipient")
                content = row.get("content") or row.get("message") or row.get("body") or ""
                date_str = row.get("date") or row.get("sent_at") or row.get("timestamp")

                record = MessageRecord(
                    conversation_id=str(row.get("conversation_id")) if pd.notna(row.get("conversation_id")) else None,
                    conversation_title=row.get("conversation_title") if pd.notna(row.get("conversation_title")) else None,
                    **{"from": _clean_name(from_name)},
                    sender_profile_url=row.get("sender_profile_url") if pd.notna(row.get("sender_profile_url")) else None,
                    to=_clean_name(to_name) if pd.notna(to_name) else None,
                    date=_parse_date(date_str) or datetime.now(),
                    subject=row.get("subject") if pd.notna(row.get("subject")) else None,
                    content=str(content) if pd.notna(content) else "",
                )
                records.append(record)
            except Exception as e:
                logger.warning(f"Skipping malformed message row: {e}")

        logger.info(f"Loaded {len(records)} messages from {filepath.name}")

    except Exception as e:
        logger.error(f"Error loading messages from {filepath}: {e}")
        raise

    return records


def _load_endorsements(filepath: Path, direction: str) -> list[EndorsementRecord]:
    """Load endorsements CSV file."""
    records = []

    try:
        df = pd.read_csv(filepath)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        for _, row in df.iterrows():
            try:
                record = EndorsementRecord(
                    endorser_name=_clean_name(row.get("endorser_name") or row.get("endorser")),
                    endorsee_name=_clean_name(row.get("endorsee_name") or row.get("endorsee")),
                    skill=row.get("skill") or row.get("skill_name"),
                    date=_parse_date(row.get("date") or row.get("endorsed_on")),
                    status=row.get("status"),
                )
                records.append(record)
            except Exception as e:
                logger.warning(f"Skipping malformed endorsement row: {e}")

        logger.info(f"Loaded {len(records)} endorsements ({direction}) from {filepath.name}")

    except Exception as e:
        logger.error(f"Error loading endorsements from {filepath}: {e}")

    return records


def _load_recommendations(filepath: Path, direction: str) -> list[RecommendationRecord]:
    """Load recommendations CSV file."""
    records = []

    try:
        df = pd.read_csv(filepath)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        for _, row in df.iterrows():
            try:
                record = RecommendationRecord(
                    first_name=_clean_name(row.get("first_name")),
                    last_name=_clean_name(row.get("last_name")),
                    company=row.get("company") if pd.notna(row.get("company")) else None,
                    job_title=row.get("job_title") or row.get("position"),
                    text=row.get("text") or row.get("recommendation"),
                    date=_parse_date(row.get("date") or row.get("created_on")),
                    status=row.get("status"),
                )
                records.append(record)
            except Exception as e:
                logger.warning(f"Skipping malformed recommendation row: {e}")

        logger.info(f"Loaded {len(records)} recommendations ({direction}) from {filepath.name}")

    except Exception as e:
        logger.error(f"Error loading recommendations from {filepath}: {e}")

    return records


def _find_file(directory: Path, patterns: list[str]) -> Optional[Path]:
    """Find a file matching one of the patterns (case-insensitive)."""
    for pattern in patterns:
        # Try exact match first
        exact_path = directory / pattern
        if exact_path.exists():
            return exact_path

        # Try case-insensitive search
        for f in directory.iterdir():
            if f.name.lower() == pattern.lower():
                return f

    return None


def load_linkedin_export(
    directory: str | Path,
    require_messages: bool = True,
) -> LinkedInExport:
    """Load LinkedIn data export from directory.

    Args:
        directory: Path to LinkedIn export directory
        require_messages: If True, raise error if Messages.csv not found

    Returns:
        LinkedInExport containing all loaded data

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If connections cannot be loaded
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    export = LinkedInExport(source_directory=str(directory))

    # Required: Connections.csv
    connections_file = _find_file(directory, ["Connections.csv", "connections.csv"])
    if connections_file:
        export.connections = _load_connections(connections_file)
        export.loaded_files.append(connections_file.name)
    else:
        raise FileNotFoundError(f"Connections.csv not found in {directory}")

    if len(export.connections) == 0:
        raise ValueError("No connections loaded from Connections.csv")

    # Required or optional: Messages.csv
    messages_file = _find_file(directory, ["Messages.csv", "messages.csv"])
    if messages_file:
        export.messages = _load_messages(messages_file)
        export.loaded_files.append(messages_file.name)
    elif require_messages:
        raise FileNotFoundError(f"Messages.csv not found in {directory}")
    else:
        export.skipped_files.append("Messages.csv")
        logger.warning("Messages.csv not found, message analysis will be limited")

    # Optional: Endorsements
    endorsements_given_file = _find_file(directory, [
        "Endorsement_Given_Info.csv",
        "Endorsements_Given.csv",
        "endorsements_given.csv",
    ])
    if endorsements_given_file:
        export.endorsements_given = _load_endorsements(endorsements_given_file, "given")
        export.loaded_files.append(endorsements_given_file.name)
    else:
        export.skipped_files.append("Endorsements_Given.csv")

    endorsements_received_file = _find_file(directory, [
        "Endorsement_Received_Info.csv",
        "Endorsements_Received.csv",
        "endorsements_received.csv",
    ])
    if endorsements_received_file:
        export.endorsements_received = _load_endorsements(endorsements_received_file, "received")
        export.loaded_files.append(endorsements_received_file.name)
    else:
        export.skipped_files.append("Endorsements_Received.csv")

    # Optional: Recommendations
    recs_given_file = _find_file(directory, [
        "Recommendations_Given.csv",
        "recommendations_given.csv",
    ])
    if recs_given_file:
        export.recommendations_given = _load_recommendations(recs_given_file, "given")
        export.loaded_files.append(recs_given_file.name)
    else:
        export.skipped_files.append("Recommendations_Given.csv")

    recs_received_file = _find_file(directory, [
        "Recommendations_Received.csv",
        "recommendations_received.csv",
    ])
    if recs_received_file:
        export.recommendations_received = _load_recommendations(recs_received_file, "received")
        export.loaded_files.append(recs_received_file.name)
    else:
        export.skipped_files.append("Recommendations_Received.csv")

    logger.info(
        f"LinkedIn export loaded: {len(export.connections)} connections, "
        f"{len(export.messages)} messages, "
        f"{len(export.loaded_files)} files loaded, "
        f"{len(export.skipped_files)} files skipped"
    )

    return export
