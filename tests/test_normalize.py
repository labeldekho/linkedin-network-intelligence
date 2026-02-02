"""
Tests for Entity Resolution and Normalization
"""

import pytest
from datetime import datetime, timedelta

from src.pipeline.normalize import (
    PersonResolver,
    normalize_network,
    _build_message_threads,
)
from src.pipeline.ingest import LinkedInExport, ConnectionRecord, MessageRecord
from src.models.entities import Interaction, InteractionType


class TestPersonResolver:
    """Tests for PersonResolver class."""

    def test_add_new_person(self):
        """Test adding a new person."""
        resolver = PersonResolver()
        person = resolver.add_person(
            first_name="John",
            last_name="Doe",
            company="Acme Corp",
        )

        assert person.first_name == "John"
        assert person.last_name == "Doe"
        assert person.company == "Acme Corp"
        assert person.id is not None

    def test_deduplicate_by_name(self):
        """Test that same name returns same person."""
        resolver = PersonResolver()

        person1 = resolver.add_person(
            first_name="John",
            last_name="Doe",
        )
        person2 = resolver.add_person(
            first_name="John",
            last_name="Doe",
            company="New Company",
        )

        assert person1.id == person2.id
        # Company should be updated
        assert person1.company == "New Company"

    def test_deduplicate_by_email(self):
        """Test that same email returns same person."""
        resolver = PersonResolver()

        person1 = resolver.add_person(
            first_name="John",
            last_name="Doe",
            email="john@example.com",
        )
        person2 = resolver.add_person(
            first_name="Johnny",  # Different name
            last_name="D",
            email="john@example.com",  # Same email
        )

        assert person1.id == person2.id

    def test_resolve_by_full_name(self):
        """Test resolving person by full name."""
        resolver = PersonResolver()
        resolver.add_person(first_name="John", last_name="Doe")

        resolved = resolver.resolve_by_name("John Doe")
        assert resolved is not None
        assert resolved.first_name == "John"

    def test_resolve_by_name_not_found(self):
        """Test resolving unknown name returns None."""
        resolver = PersonResolver()
        resolver.add_person(first_name="John", last_name="Doe")

        resolved = resolver.resolve_by_name("Jane Smith")
        assert resolved is None

    def test_normalize_name_case_insensitive(self):
        """Test that name matching is case-insensitive."""
        resolver = PersonResolver()
        resolver.add_person(first_name="John", last_name="Doe")

        resolved = resolver.resolve_by_name("JOHN DOE")
        assert resolved is not None

    def test_get_all_people(self):
        """Test getting all resolved people."""
        resolver = PersonResolver()
        resolver.add_person(first_name="John", last_name="Doe")
        resolver.add_person(first_name="Jane", last_name="Smith")

        people = resolver.get_all_people()
        assert len(people) == 2


class TestBuildMessageThreads:
    """Tests for message thread building."""

    def test_single_thread(self):
        """Test building a single thread from sequential messages."""
        now = datetime.now()
        interactions = [
            Interaction(
                person_id="p1",
                type=InteractionType.MESSAGE_SENT,
                timestamp=now - timedelta(days=5),
            ),
            Interaction(
                person_id="p1",
                type=InteractionType.MESSAGE_RECEIVED,
                timestamp=now - timedelta(days=4),
            ),
            Interaction(
                person_id="p1",
                type=InteractionType.MESSAGE_SENT,
                timestamp=now - timedelta(days=3),
            ),
        ]

        threads = _build_message_threads(interactions, thread_break_days=7)

        assert len(threads) == 1
        assert threads[0].message_count == 3

    def test_multiple_threads_with_gap(self):
        """Test that gap in messages creates separate threads."""
        now = datetime.now()
        interactions = [
            Interaction(
                person_id="p1",
                type=InteractionType.MESSAGE_SENT,
                timestamp=now - timedelta(days=30),
            ),
            Interaction(
                person_id="p1",
                type=InteractionType.MESSAGE_RECEIVED,
                timestamp=now - timedelta(days=29),
            ),
            # 20 day gap
            Interaction(
                person_id="p1",
                type=InteractionType.MESSAGE_SENT,
                timestamp=now - timedelta(days=5),
            ),
        ]

        threads = _build_message_threads(interactions, thread_break_days=7)

        assert len(threads) == 2
        assert threads[0].message_count == 2
        assert threads[1].message_count == 1

    def test_empty_interactions(self):
        """Test handling empty interactions list."""
        threads = _build_message_threads([], thread_break_days=7)
        assert threads == []

    def test_non_message_interactions_filtered(self):
        """Test that non-message interactions are filtered."""
        now = datetime.now()
        interactions = [
            Interaction(
                person_id="p1",
                type=InteractionType.CONNECTION,
                timestamp=now - timedelta(days=30),
            ),
            Interaction(
                person_id="p1",
                type=InteractionType.ENDORSEMENT_GIVEN,
                timestamp=now - timedelta(days=20),
            ),
        ]

        threads = _build_message_threads(interactions, thread_break_days=7)
        assert threads == []


class TestNormalizeNetwork:
    """Tests for network normalization."""

    def test_normalize_connections_only(self):
        """Test normalizing export with only connections."""
        export = LinkedInExport(
            connections=[
                ConnectionRecord(
                    first_name="John",
                    last_name="Doe",
                    company="Acme Corp",
                    connected_on=datetime(2024, 1, 15),
                ),
                ConnectionRecord(
                    first_name="Jane",
                    last_name="Smith",
                    company="Tech Inc",
                    connected_on=datetime(2024, 2, 20),
                ),
            ],
        )

        snapshot = normalize_network(export)

        assert len(snapshot.people) == 2
        assert len(snapshot.relationships) == 2
        assert snapshot.total_connections == 2

    def test_normalize_with_messages(self):
        """Test normalizing export with connections and messages."""
        now = datetime.now()

        export = LinkedInExport(
            connections=[
                ConnectionRecord(
                    first_name="John",
                    last_name="Doe",
                    company="Acme Corp",
                    connected_on=datetime(2024, 1, 15),
                ),
            ],
            messages=[
                MessageRecord(
                    **{"from": "John Doe"},
                    date=now - timedelta(days=5),
                    content="Hello!",
                ),
            ],
        )

        snapshot = normalize_network(export)

        # John Doe should have both connection and message interactions
        john_rel = None
        for rel in snapshot.relationships.values():
            if rel.person.first_name == "John":
                john_rel = rel
                break

        assert john_rel is not None
        assert len(john_rel.interactions) == 2  # Connection + message

    def test_normalize_creates_person_from_message(self):
        """Test that messages create new people if not in connections."""
        export = LinkedInExport(
            connections=[
                ConnectionRecord(
                    first_name="John",
                    last_name="Doe",
                ),
            ],
            messages=[
                MessageRecord(
                    **{"from": "Unknown Person"},
                    date=datetime.now(),
                    content="Hello!",
                ),
            ],
        )

        snapshot = normalize_network(export)

        # Should have both John and Unknown Person
        assert len(snapshot.people) == 2
