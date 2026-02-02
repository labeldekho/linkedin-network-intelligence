"""
Tests for Reciprocity Ledger
"""

import pytest
from datetime import datetime, timedelta

from src.models.reciprocity import (
    ReciprocityLedger,
    ReciprocityBalance,
    ReciprocityStatus,
)
from src.models.entities import (
    Person,
    Interaction,
    InteractionType,
    Relationship,
    NetworkSnapshot,
)


class TestReciprocityLedger:
    """Tests for ReciprocityLedger class."""

    @pytest.fixture
    def ledger(self):
        """Create a ledger with default settings."""
        return ReciprocityLedger()

    @pytest.fixture
    def person(self):
        """Create a test person."""
        return Person(
            id="test",
            first_name="John",
            last_name="Doe",
        )

    def test_balanced_relationship(self, ledger, person):
        """Test that relationship with no significant interactions is balanced."""
        relationship = Relationship(
            person=person,
            interactions=[
                Interaction(
                    person_id="test",
                    type=InteractionType.MESSAGE_SENT,
                    timestamp=datetime.now(),
                ),
            ],
        )

        balance = ledger.calculate_balance(relationship)

        assert balance.balance == 0
        assert balance.status == ReciprocityStatus.BALANCED

    def test_credit_relationship(self, ledger, person):
        """Test relationship where you gave more."""
        relationship = Relationship(
            person=person,
            interactions=[
                Interaction(
                    person_id="test",
                    type=InteractionType.RECOMMENDATION_WRITTEN,
                    timestamp=datetime.now(),
                ),
                Interaction(
                    person_id="test",
                    type=InteractionType.ENDORSEMENT_GIVEN,
                    timestamp=datetime.now(),
                ),
            ],
        )

        balance = ledger.calculate_balance(relationship)

        assert balance.balance == 12  # 10 + 2
        assert balance.status == ReciprocityStatus.CREDIT

    def test_strong_credit_relationship(self, ledger, person):
        """Test relationship with strong credit."""
        relationship = Relationship(
            person=person,
            interactions=[
                Interaction(
                    person_id="test",
                    type=InteractionType.RECOMMENDATION_WRITTEN,
                    timestamp=datetime.now(),
                ),
                Interaction(
                    person_id="test",
                    type=InteractionType.RECOMMENDATION_WRITTEN,
                    timestamp=datetime.now(),
                ),
            ],
        )

        balance = ledger.calculate_balance(relationship)

        assert balance.balance == 20
        assert balance.status == ReciprocityStatus.STRONG_CREDIT

    def test_debit_relationship(self, ledger, person):
        """Test relationship where you received more."""
        relationship = Relationship(
            person=person,
            interactions=[
                Interaction(
                    person_id="test",
                    type=InteractionType.RECOMMENDATION_RECEIVED,
                    timestamp=datetime.now(),
                ),
            ],
        )

        balance = ledger.calculate_balance(relationship)

        assert balance.balance == -10
        assert balance.status == ReciprocityStatus.DEBIT

    def test_strong_debit_relationship(self, ledger, person):
        """Test relationship with strong debit."""
        relationship = Relationship(
            person=person,
            interactions=[
                Interaction(
                    person_id="test",
                    type=InteractionType.RECOMMENDATION_RECEIVED,
                    timestamp=datetime.now(),
                ),
                Interaction(
                    person_id="test",
                    type=InteractionType.RECOMMENDATION_RECEIVED,
                    timestamp=datetime.now(),
                ),
            ],
        )

        balance = ledger.calculate_balance(relationship)

        assert balance.balance == -20
        assert balance.status == ReciprocityStatus.STRONG_DEBIT

    def test_balance_breakdown(self, ledger, person):
        """Test that balance includes proper breakdown."""
        relationship = Relationship(
            person=person,
            interactions=[
                Interaction(
                    person_id="test",
                    type=InteractionType.RECOMMENDATION_WRITTEN,
                    timestamp=datetime.now(),
                ),
                Interaction(
                    person_id="test",
                    type=InteractionType.ENDORSEMENT_RECEIVED,
                    timestamp=datetime.now(),
                ),
            ],
        )

        balance = ledger.calculate_balance(relationship)

        assert "recommendation_written" in balance.given
        assert "endorsement_received" in balance.received
        assert balance.total_given == 10
        assert balance.total_received == 2

    def test_network_balances(self, ledger, sample_network_snapshot):
        """Test calculating balances for entire network."""
        snapshot, balances = ledger.calculate_network_balances(sample_network_snapshot)

        assert len(balances) == len(sample_network_snapshot.relationships)

        # All relationships should have reciprocity balance
        for rel in snapshot.relationships.values():
            assert rel.reciprocity_balance is not None

    def test_get_credit_relationships(self, ledger, sample_network_snapshot):
        """Test filtering credit relationships."""
        snapshot, _ = ledger.calculate_network_balances(sample_network_snapshot)
        credit_rels = ledger.get_credit_relationships(snapshot, min_balance=5.0)

        for rel in credit_rels:
            assert rel.reciprocity_balance >= 5.0

    def test_get_debit_relationships(self, ledger, sample_network_snapshot):
        """Test filtering debit relationships."""
        snapshot, _ = ledger.calculate_network_balances(sample_network_snapshot)
        debit_rels = ledger.get_debit_relationships(snapshot, max_balance=-5.0)

        for rel in debit_rels:
            assert rel.reciprocity_balance <= -5.0

    def test_get_summary(self, ledger, sample_network_snapshot):
        """Test getting summary statistics."""
        _, balances = ledger.calculate_network_balances(sample_network_snapshot)
        summary = ledger.get_summary(balances)

        assert "total_relationships" in summary
        assert "by_status" in summary
        assert "avg_balance" in summary
        assert "total_given" in summary
        assert "total_received" in summary

        assert summary["total_relationships"] == len(balances)

    def test_empty_summary(self, ledger):
        """Test summary with no balances."""
        summary = ledger.get_summary([])

        assert summary["total_relationships"] == 0
        assert summary["avg_balance"] == 0


class TestReciprocityBalance:
    """Tests for ReciprocityBalance model."""

    def test_is_credit(self):
        """Test is_credit property."""
        balance = ReciprocityBalance(
            person_id="test",
            person_name="Test",
            balance=10,
            status=ReciprocityStatus.CREDIT,
        )
        assert balance.is_credit
        assert not balance.is_debit

    def test_is_debit(self):
        """Test is_debit property."""
        balance = ReciprocityBalance(
            person_id="test",
            person_name="Test",
            balance=-10,
            status=ReciprocityStatus.DEBIT,
        )
        assert balance.is_debit
        assert not balance.is_credit


class TestCustomScores:
    """Tests for custom reciprocity scores."""

    def test_custom_recommendation_score(self):
        """Test using custom recommendation score."""
        ledger = ReciprocityLedger(
            scores={"recommendation_written": 20}
        )

        person = Person(id="test", first_name="Test", last_name="User")
        relationship = Relationship(
            person=person,
            interactions=[
                Interaction(
                    person_id="test",
                    type=InteractionType.RECOMMENDATION_WRITTEN,
                    timestamp=datetime.now(),
                ),
            ],
        )

        balance = ledger.calculate_balance(relationship)
        assert balance.balance == 20

    def test_custom_thresholds(self):
        """Test using custom thresholds."""
        ledger = ReciprocityLedger(
            thresholds={
                "strong_credit": 25,
                "credit": 10,
            }
        )

        # 12 points should now be credit (not strong_credit with default 15)
        person = Person(id="test", first_name="Test", last_name="User")
        relationship = Relationship(
            person=person,
            interactions=[
                Interaction(
                    person_id="test",
                    type=InteractionType.RECOMMENDATION_WRITTEN,
                    timestamp=datetime.now(),
                ),
                Interaction(
                    person_id="test",
                    type=InteractionType.ENDORSEMENT_GIVEN,
                    timestamp=datetime.now(),
                ),
            ],
        )

        balance = ledger.calculate_balance(relationship)
        assert balance.balance == 12
        assert balance.status == ReciprocityStatus.CREDIT
