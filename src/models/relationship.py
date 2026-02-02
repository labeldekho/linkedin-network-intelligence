"""
Relationship Strength Calculator

Implements decay-based relationship strength scoring.
"""

import logging
import math
from datetime import datetime
from typing import Optional

from src.models.entities import (
    Interaction,
    InteractionType,
    NetworkSnapshot,
    Relationship,
)

logger = logging.getLogger(__name__)


class RelationshipStrengthCalculator:
    """Calculates relationship strength using decay-based formula.

    Formula:
        strength = sum(interaction_weight * depth_multiplier * decay_factor)
        decay_factor = 0.5^(days_since / half_life)
    """

    DEFAULT_WEIGHTS = {
        InteractionType.MESSAGE_SENT: 1.0,
        InteractionType.MESSAGE_RECEIVED: 1.0,
        InteractionType.RECOMMENDATION_WRITTEN: 5.0,
        InteractionType.RECOMMENDATION_RECEIVED: 5.0,
        InteractionType.ENDORSEMENT_GIVEN: 0.5,
        InteractionType.ENDORSEMENT_RECEIVED: 0.5,
        InteractionType.CONNECTION: 0.1,
    }

    def __init__(
        self,
        decay_half_life_days: int = 180,
        minimum_strength: float = 0.05,
        depth_multiplier_max: float = 2.0,
        institutional_multiplier: float = 1.5,
        interaction_weights: Optional[dict[str, float]] = None,
    ):
        """Initialize calculator with configuration.

        Args:
            decay_half_life_days: Days until relationship strength halves
            minimum_strength: Floor value (never decays below this)
            depth_multiplier_max: Maximum depth multiplier
            institutional_multiplier: Bonus for shared institution
            interaction_weights: Custom weights per interaction type
        """
        self.decay_half_life_days = decay_half_life_days
        self.minimum_strength = minimum_strength
        self.depth_multiplier_max = depth_multiplier_max
        self.institutional_multiplier = institutional_multiplier

        # Build weights dict
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if interaction_weights:
            for key, value in interaction_weights.items():
                # Handle string keys
                if isinstance(key, str):
                    try:
                        int_type = InteractionType(key)
                        self.weights[int_type] = value
                    except ValueError:
                        logger.warning(f"Unknown interaction type: {key}")
                else:
                    self.weights[key] = value

    def _calculate_decay(self, days_since: int) -> float:
        """Calculate decay factor for given days since interaction.

        Uses exponential decay: 0.5^(days / half_life)
        """
        if days_since <= 0:
            return 1.0

        decay = math.pow(0.5, days_since / self.decay_half_life_days)
        return max(decay, self.minimum_strength)

    def _get_depth_multiplier(self, interaction: Interaction) -> float:
        """Get depth multiplier for an interaction.

        Uses LLM-assessed depth score if available, otherwise 1.0.
        """
        if interaction.depth_score is not None:
            # Scale depth score (0-1) to multiplier (1-max)
            return 1.0 + (interaction.depth_score * (self.depth_multiplier_max - 1.0))
        return 1.0

    def calculate_interaction_score(
        self,
        interaction: Interaction,
        reference_date: Optional[datetime] = None,
    ) -> float:
        """Calculate score contribution from a single interaction.

        Args:
            interaction: The interaction to score
            reference_date: Date to calculate decay from (default: now)

        Returns:
            Score contribution from this interaction
        """
        reference_date = reference_date or datetime.now()

        # Base weight for interaction type
        base_weight = self.weights.get(interaction.type, 1.0)

        # Depth multiplier
        depth_mult = self._get_depth_multiplier(interaction)

        # Decay factor
        days_since = (reference_date - interaction.timestamp).days
        decay = self._calculate_decay(days_since)

        return base_weight * depth_mult * decay

    def calculate_relationship_strength(
        self,
        relationship: Relationship,
        reference_date: Optional[datetime] = None,
        shared_institution: bool = False,
    ) -> float:
        """Calculate total relationship strength.

        Args:
            relationship: The relationship to score
            reference_date: Date to calculate decay from (default: now)
            shared_institution: Whether you share institution with person

        Returns:
            Total relationship strength score
        """
        reference_date = reference_date or datetime.now()

        if not relationship.interactions:
            return self.minimum_strength

        # Sum all interaction scores
        total_score = sum(
            self.calculate_interaction_score(interaction, reference_date)
            for interaction in relationship.interactions
        )

        # Apply institutional multiplier
        if shared_institution:
            total_score *= self.institutional_multiplier

        return max(total_score, self.minimum_strength)

    def calculate_network_strengths(
        self,
        snapshot: NetworkSnapshot,
        reference_date: Optional[datetime] = None,
        shared_institutions: Optional[set[str]] = None,
    ) -> NetworkSnapshot:
        """Calculate strength scores for all relationships in snapshot.

        Args:
            snapshot: Network snapshot to process
            reference_date: Date to calculate decay from
            shared_institutions: Company names considered shared institutions

        Returns:
            Updated NetworkSnapshot with strength scores
        """
        reference_date = reference_date or datetime.now()
        shared_institutions = shared_institutions or set()

        for person_id, relationship in snapshot.relationships.items():
            # Check for shared institution
            shared = False
            if relationship.person.company:
                if relationship.person.company.lower() in {
                    s.lower() for s in shared_institutions
                }:
                    shared = True

            # Calculate strength
            strength = self.calculate_relationship_strength(
                relationship,
                reference_date=reference_date,
                shared_institution=shared,
            )
            relationship.strength_score = strength

            # Update dormancy status
            if relationship.days_since_last_interaction is not None:
                relationship.is_dormant = relationship.days_since_last_interaction >= 90

        logger.info(f"Calculated strength scores for {len(snapshot.relationships)} relationships")

        return snapshot

    def get_strength_breakdown(
        self,
        relationship: Relationship,
        reference_date: Optional[datetime] = None,
    ) -> dict:
        """Get detailed breakdown of strength score.

        Args:
            relationship: The relationship to analyze
            reference_date: Date to calculate decay from

        Returns:
            Dictionary with breakdown by interaction type
        """
        reference_date = reference_date or datetime.now()

        breakdown = {
            "total": 0.0,
            "by_type": {},
            "by_interaction": [],
        }

        for interaction in relationship.interactions:
            score = self.calculate_interaction_score(interaction, reference_date)
            breakdown["total"] += score

            type_name = interaction.type.value
            if type_name not in breakdown["by_type"]:
                breakdown["by_type"][type_name] = 0.0
            breakdown["by_type"][type_name] += score

            days_since = (reference_date - interaction.timestamp).days
            breakdown["by_interaction"].append({
                "type": type_name,
                "date": interaction.timestamp.isoformat(),
                "days_ago": days_since,
                "base_weight": self.weights.get(interaction.type, 1.0),
                "depth_multiplier": self._get_depth_multiplier(interaction),
                "decay_factor": self._calculate_decay(days_since),
                "score": score,
            })

        return breakdown
