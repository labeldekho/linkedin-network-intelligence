"""
Pytest Configuration and Shared Fixtures
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from src.models.entities import (
    Person,
    Interaction,
    InteractionType,
    Relationship,
    MessageThread,
    NetworkSnapshot,
)
from src.pipeline.ingest import (
    ConnectionRecord,
    MessageRecord,
    LinkedInExport,
)


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_person() -> Person:
    """Create a sample person for testing."""
    return Person(
        id="test_person_001",
        first_name="Jane",
        last_name="Smith",
        company="Acme Corp",
        position="Senior Engineer",
        email="jane.smith@example.com",
        connected_on=datetime(2023, 1, 15),
    )


@pytest.fixture
def sample_interaction() -> Interaction:
    """Create a sample interaction for testing."""
    return Interaction(
        person_id="test_person_001",
        type=InteractionType.MESSAGE_SENT,
        timestamp=datetime(2024, 6, 15),
        depth_score=0.7,
        content_summary="Discussion about project collaboration",
    )


@pytest.fixture
def sample_interactions() -> list[Interaction]:
    """Create a list of sample interactions."""
    now = datetime.now()
    return [
        Interaction(
            person_id="test_person_001",
            type=InteractionType.CONNECTION,
            timestamp=now - timedelta(days=365),
        ),
        Interaction(
            person_id="test_person_001",
            type=InteractionType.MESSAGE_SENT,
            timestamp=now - timedelta(days=180),
            depth_score=0.5,
        ),
        Interaction(
            person_id="test_person_001",
            type=InteractionType.MESSAGE_RECEIVED,
            timestamp=now - timedelta(days=175),
            depth_score=0.6,
        ),
        Interaction(
            person_id="test_person_001",
            type=InteractionType.RECOMMENDATION_WRITTEN,
            timestamp=now - timedelta(days=90),
        ),
        Interaction(
            person_id="test_person_001",
            type=InteractionType.MESSAGE_SENT,
            timestamp=now - timedelta(days=30),
            depth_score=0.8,
        ),
    ]


@pytest.fixture
def sample_relationship(sample_person, sample_interactions) -> Relationship:
    """Create a sample relationship with interactions."""
    return Relationship(
        person=sample_person,
        interactions=sample_interactions,
    )


@pytest.fixture
def sample_network_snapshot() -> NetworkSnapshot:
    """Create a sample network snapshot for testing."""
    now = datetime.now()

    people = {}
    relationships = {}

    # Create several people with varying interaction patterns
    test_cases = [
        {
            "first_name": "Alice",
            "last_name": "Johnson",
            "company": "TechCorp",
            "position": "CTO",
            "interactions": [
                (InteractionType.MESSAGE_SENT, 30, 0.8),
                (InteractionType.MESSAGE_RECEIVED, 28, 0.7),
                (InteractionType.RECOMMENDATION_WRITTEN, 60, None),
            ],
        },
        {
            "first_name": "Bob",
            "last_name": "Williams",
            "company": "StartupXYZ",
            "position": "Founder",
            "interactions": [
                (InteractionType.CONNECTION, 365, None),
                (InteractionType.MESSAGE_SENT, 200, 0.3),
            ],
        },
        {
            "first_name": "Carol",
            "last_name": "Davis",
            "company": "BigCo",
            "position": "VP Engineering",
            "interactions": [
                (InteractionType.MESSAGE_RECEIVED, 10, 0.9),
                (InteractionType.ENDORSEMENT_RECEIVED, 15, None),
                (InteractionType.RECOMMENDATION_RECEIVED, 20, None),
            ],
        },
        {
            "first_name": "David",
            "last_name": "Brown",
            "company": "Anthropic",
            "position": "Engineer",
            "interactions": [
                (InteractionType.CONNECTION, 180, None),
            ],
        },
    ]

    for i, case in enumerate(test_cases):
        person_id = f"person_{i:03d}"

        person = Person(
            id=person_id,
            first_name=case["first_name"],
            last_name=case["last_name"],
            company=case["company"],
            position=case["position"],
            connected_on=now - timedelta(days=365),
        )
        people[person_id] = person

        interactions = []
        for int_type, days_ago, depth in case["interactions"]:
            interaction = Interaction(
                person_id=person_id,
                type=int_type,
                timestamp=now - timedelta(days=days_ago),
                depth_score=depth,
            )
            interactions.append(interaction)

        relationship = Relationship(
            person=person,
            interactions=interactions,
        )
        relationships[person_id] = relationship

    return NetworkSnapshot(
        people=people,
        relationships=relationships,
        source_files=["Connections.csv", "Messages.csv"],
    )


@pytest.fixture
def sample_connection_records() -> list[ConnectionRecord]:
    """Create sample connection records."""
    return [
        ConnectionRecord(
            first_name="John",
            last_name="Doe",
            company="Example Inc",
            position="Developer",
            email_address="john.doe@example.com",
            connected_on=datetime(2023, 5, 10),
        ),
        ConnectionRecord(
            first_name="Jane",
            last_name="Smith",
            company="Tech Corp",
            position="Manager",
            connected_on=datetime(2023, 8, 22),
        ),
        ConnectionRecord(
            first_name="Bob",
            last_name="Johnson",
            company="Startup XYZ",
            position="Founder",
            email_address="bob@startup.xyz",
            connected_on=datetime(2024, 1, 5),
        ),
    ]


@pytest.fixture
def sample_message_records() -> list[MessageRecord]:
    """Create sample message records."""
    return [
        MessageRecord(
            conversation_id="conv_001",
            **{"from": "John Doe"},
            to="You",
            date=datetime(2024, 5, 15, 10, 30),
            content="Hi, would love to connect about the project.",
        ),
        MessageRecord(
            conversation_id="conv_001",
            **{"from": "You"},
            to="John Doe",
            date=datetime(2024, 5, 15, 14, 0),
            content="Sure, let's schedule a call!",
        ),
        MessageRecord(
            conversation_id="conv_002",
            **{"from": "Jane Smith"},
            to="You",
            date=datetime(2024, 6, 1, 9, 0),
            content="Congrats on the new role!",
        ),
    ]


@pytest.fixture
def sample_linkedin_export(
    sample_connection_records,
    sample_message_records,
) -> LinkedInExport:
    """Create a sample LinkedIn export."""
    return LinkedInExport(
        connections=sample_connection_records,
        messages=sample_message_records,
        source_directory="/test/data",
        loaded_files=["Connections.csv", "Messages.csv"],
    )
