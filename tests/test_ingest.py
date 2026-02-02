"""
Tests for Data Ingestion Pipeline
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, mock_open
import pandas as pd
import io

from src.pipeline.ingest import (
    load_linkedin_export,
    _parse_date,
    _clean_name,
    _load_connections,
    ConnectionRecord,
    MessageRecord,
    LinkedInExport,
)


class TestParseDateFunction:
    """Tests for date parsing."""

    def test_parse_linkedin_format(self):
        """Test parsing LinkedIn's date format."""
        result = _parse_date("15 Jan 2024")
        assert result == datetime(2024, 1, 15)

    def test_parse_iso_format(self):
        """Test parsing ISO date format."""
        result = _parse_date("2024-01-15")
        assert result == datetime(2024, 1, 15)

    def test_parse_us_format(self):
        """Test parsing US date format."""
        result = _parse_date("01/15/2024")
        assert result == datetime(2024, 1, 15)

    def test_parse_none(self):
        """Test parsing None returns None."""
        assert _parse_date(None) is None

    def test_parse_invalid(self):
        """Test parsing invalid date returns None."""
        result = _parse_date("not a date")
        assert result is None


class TestCleanNameFunction:
    """Tests for name cleaning."""

    def test_clean_normal_name(self):
        """Test cleaning a normal name."""
        assert _clean_name("John Doe") == "John Doe"

    def test_clean_with_whitespace(self):
        """Test cleaning name with extra whitespace."""
        assert _clean_name("  John Doe  ") == "John Doe"

    def test_clean_none(self):
        """Test cleaning None returns empty string."""
        assert _clean_name(None) == ""


class TestConnectionRecord:
    """Tests for ConnectionRecord model."""

    def test_create_connection_record(self):
        """Test creating a connection record."""
        record = ConnectionRecord(
            first_name="Jane",
            last_name="Smith",
            company="Acme Corp",
            position="Engineer",
            connected_on=datetime(2024, 1, 15),
        )

        assert record.first_name == "Jane"
        assert record.last_name == "Smith"
        assert record.company == "Acme Corp"
        assert record.email_address is None

    def test_connection_record_optional_fields(self):
        """Test connection record with optional fields."""
        record = ConnectionRecord(
            first_name="John",
            last_name="Doe",
        )

        assert record.company is None
        assert record.position is None
        assert record.connected_on is None


class TestMessageRecord:
    """Tests for MessageRecord model."""

    def test_create_message_record(self):
        """Test creating a message record."""
        record = MessageRecord(
            **{"from": "John Doe"},
            date=datetime(2024, 6, 15, 10, 30),
            content="Hello, world!",
        )

        assert record.from_name == "John Doe"
        assert record.content == "Hello, world!"
        assert record.to_name is None


class TestLinkedInExport:
    """Tests for LinkedInExport model."""

    def test_empty_export(self):
        """Test creating empty export."""
        export = LinkedInExport()

        assert export.connections == []
        assert export.messages == []
        assert not export.has_connections
        assert not export.has_messages

    def test_export_with_data(self, sample_linkedin_export):
        """Test export with data."""
        assert sample_linkedin_export.has_connections
        assert sample_linkedin_export.has_messages
        assert len(sample_linkedin_export.connections) == 3
        assert len(sample_linkedin_export.messages) == 3


class TestLoadLinkedInExport:
    """Tests for loading LinkedIn export from directory."""

    def test_load_missing_directory(self, tmp_path):
        """Test loading from non-existent directory raises error."""
        with pytest.raises(FileNotFoundError):
            load_linkedin_export(tmp_path / "nonexistent")

    def test_load_missing_connections_file(self, tmp_path):
        """Test loading without Connections.csv raises error."""
        with pytest.raises(FileNotFoundError, match="Connections.csv"):
            load_linkedin_export(tmp_path)

    def test_load_with_connections_only(self, tmp_path, fixtures_dir):
        """Test loading with only Connections.csv."""
        # Create a minimal connections file
        connections_content = """Notes:
Some notes here
More notes

First Name,Last Name,Email Address,Company,Position,Connected On
John,Doe,john@example.com,Acme Corp,Engineer,15 Jan 2024
Jane,Smith,,Tech Inc,Manager,20 Feb 2024
"""
        (tmp_path / "Connections.csv").write_text(connections_content)

        export = load_linkedin_export(tmp_path, require_messages=False)

        assert len(export.connections) == 2
        assert export.connections[0].first_name == "John"
        assert export.connections[1].company == "Tech Inc"
        assert "Messages.csv" in export.skipped_files

    def test_load_with_connections_and_messages(self, tmp_path):
        """Test loading with both files."""
        # Create connections file
        connections_content = """Notes:
Some notes here
More notes

First Name,Last Name,Email Address,Company,Position,Connected On
John,Doe,john@example.com,Acme Corp,Engineer,15 Jan 2024
"""
        (tmp_path / "Connections.csv").write_text(connections_content)

        # Create messages file
        messages_content = """CONVERSATION ID,CONVERSATION TITLE,FROM,SENDER PROFILE URL,TO,DATE,SUBJECT,CONTENT
conv1,Chat with John,John Doe,,You,2024-06-15 10:30:00,,Hello there!
"""
        (tmp_path / "Messages.csv").write_text(messages_content)

        export = load_linkedin_export(tmp_path, require_messages=True)

        assert len(export.connections) == 1
        assert len(export.messages) == 1
        assert "Connections.csv" in export.loaded_files
        assert "Messages.csv" in export.loaded_files
