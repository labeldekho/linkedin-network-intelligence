"""
Tests for Relationship Strength Calculator
"""

import pytest
from datetime import datetime, timedelta
import math

from src.models.relationship import RelationshipStrengthCalculator
from src.models.entities import (
    Person,
    Interaction,
    InteractionType,
    Relationship,
    NetworkSnapshot,
)


class TestRelationshipStrengthCalculator:
    """Tests for RelationshipStrengthCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator with default settings."""
        return RelationshipStrengthCalculator(
            decay_half_life_days=180,
            minimum_strength=0.05,
            depth_multiplier_max=2.0,
        )

    @pytest.fixture
    def recent_interaction(self):
        """Create a recent interaction."""
        return Interaction(
            person_id="test",
            type=InteractionType.MESSAGE_SENT,
            timestamp=datetime.now() - timedelta(days=1),
        )

    @pytest.fixture
    def old_interaction(self):
        """Create an old interaction."""
        return Interaction(
            person_id="test",
            type=InteractionType.MESSAGE_SENT,
            timestamp=datetime.now() - timedelta(days=365),
        )

    def test_decay_at_half_life(self, calculator):
        """Test that decay is 0.5 at half-life."""
        decay = calculator._calculate_decay(180)
        assert abs(decay - 0.5) < 0.01

    def test_decay_recent(self, calculator):
        """Test that recent interactions have minimal decay."""
        decay = calculator._calculate_decay(1)
        assert decay > 0.99

    def test_decay_very_old(self, calculator):
        """Test that very old interactions decay to minimum."""
        decay = calculator._calculate_decay(3650)  # 10 years
        assert decay == calculator.minimum_strength

    def test_depth_multiplier_default(self, calculator):
        """Test default depth multiplier without score."""
        interaction = Interaction(
            person_id="test",
            type=InteractionType.MESSAGE_SENT,
            timestamp=datetime.now(),
            depth_score=None,
        )
        mult = calculator._get_depth_multiplier(interaction)
        assert mult == 1.0

    def test_depth_multiplier_with_score(self, calculator):
        """Test depth multiplier with high score."""
        interaction = Interaction(
            person_id="test",
            type=InteractionType.MESSAGE_SENT,
            timestamp=datetime.now(),
            depth_score=1.0,
        )
        mult = calculator._get_depth_multiplier(interaction)
        assert mult == calculator.depth_multiplier_max

    def test_interaction_score_recent(self, calculator, recent_interaction):
        """Test scoring a recent interaction."""
        score = calculator.calculate_interaction_score(recent_interaction)
        # Recent message should be close to base weight (1.0)
        assert 0.9 < score < 1.1

    def test_interaction_score_old(self, calculator, old_interaction):
        """Test scoring an old interaction."""
        score = calculator.calculate_interaction_score(old_interaction)
        # Old message should be significantly decayed
        assert score < 0.5

    def test_recommendation_weight(self, calculator):
        """Test that recommendations have higher weight."""
        msg_interaction = Interaction(
            person_id="test",
            type=InteractionType.MESSAGE_SENT,
            timestamp=datetime.now(),
        )
        rec_interaction = Interaction(
            person_id="test",
            type=InteractionType.RECOMMENDATION_WRITTEN,
            timestamp=datetime.now(),
        )

        msg_score = calculator.calculate_interaction_score(msg_interaction)
        rec_score = calculator.calculate_interaction_score(rec_interaction)

        assert rec_score > msg_score
        assert rec_score / msg_score == pytest.approx(5.0, rel=0.1)

    def test_relationship_strength_empty(self, calculator, sample_person):
        """Test relationship strength with no interactions."""
        relationship = Relationship(person=sample_person, interactions=[])
        strength = calculator.calculate_relationship_strength(relationship)
        assert strength == calculator.minimum_strength

    def test_relationship_strength_with_interactions(
        self, calculator, sample_relationship
    ):
        """Test relationship strength calculation."""
        strength = calculator.calculate_relationship_strength(sample_relationship)
        assert strength > 0
        assert strength > calculator.minimum_strength

    def test_institutional_multiplier(self, calculator, sample_relationship):
        """Test that shared institution increases strength."""
        normal_strength = calculator.calculate_relationship_strength(
            sample_relationship,
            shared_institution=False,
        )
        institutional_strength = calculator.calculate_relationship_strength(
            sample_relationship,
            shared_institution=True,
        )

        assert institutional_strength > normal_strength
        assert institutional_strength / normal_strength == pytest.approx(
            calculator.institutional_multiplier, rel=0.01
        )

    def test_network_strengths(self, calculator, sample_network_snapshot):
        """Test calculating strengths for entire network."""
        snapshot = calculator.calculate_network_strengths(sample_network_snapshot)

        # All relationships should have strength scores
        scores = []
        for rel in snapshot.relationships.values():
            assert rel.strength_score >= calculator.minimum_strength
            scores.append(rel.strength_score)

        # At least some scores should be above minimum (active relationships)
        assert any(s > calculator.minimum_strength for s in scores)

    def test_strength_breakdown(self, calculator, sample_relationship):
        """Test getting detailed strength breakdown."""
        breakdown = calculator.get_strength_breakdown(sample_relationship)

        assert "total" in breakdown
        assert "by_type" in breakdown
        assert "by_interaction" in breakdown
        assert breakdown["total"] > 0
        assert len(breakdown["by_interaction"]) == len(sample_relationship.interactions)


class TestCustomWeights:
    """Tests for custom interaction weights."""

    def test_custom_message_weight(self):
        """Test using custom message weight."""
        calculator = RelationshipStrengthCalculator(
            interaction_weights={"message_sent": 2.0}
        )

        interaction = Interaction(
            person_id="test",
            type=InteractionType.MESSAGE_SENT,
            timestamp=datetime.now(),
        )

        score = calculator.calculate_interaction_score(interaction)
        assert 1.9 < score < 2.1  # Should be around 2.0

    def test_zero_weight_interaction(self):
        """Test that zero weight returns zero score."""
        calculator = RelationshipStrengthCalculator(
            interaction_weights={"message_sent": 0.0}
        )

        interaction = Interaction(
            person_id="test",
            type=InteractionType.MESSAGE_SENT,
            timestamp=datetime.now(),
        )

        score = calculator.calculate_interaction_score(interaction)
        assert score == 0.0
