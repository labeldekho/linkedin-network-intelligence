"""
Reciprocity Ledger

Tracks social capital balance in relationships.
"""

import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.models.entities import (
    Interaction,
    InteractionType,
    NetworkSnapshot,
    Relationship,
)

logger = logging.getLogger(__name__)


class ReciprocityStatus(str, Enum):
    """Reciprocity balance status."""
    STRONG_CREDIT = "strong_credit"  # You've given significantly more
    CREDIT = "credit"                # You've given more
    BALANCED = "balanced"            # Roughly balanced
    DEBIT = "debit"                  # You've received more
    STRONG_DEBIT = "strong_debit"    # You've received significantly more


class ReciprocityBalance(BaseModel):
    """Reciprocity balance for a single relationship."""
    person_id: str
    person_name: str
    balance: float = 0.0
    status: ReciprocityStatus = ReciprocityStatus.BALANCED

    # Breakdown
    given: dict[str, int] = Field(default_factory=dict)
    received: dict[str, int] = Field(default_factory=dict)
    total_given: float = 0.0
    total_received: float = 0.0

    @property
    def is_credit(self) -> bool:
        return self.balance > 0

    @property
    def is_debit(self) -> bool:
        return self.balance < 0


class ReciprocityLedger:
    """Tracks social capital balance across relationships.

    Positive balance = you've given more value
    Negative balance = you've received more value
    """

    DEFAULT_SCORES = {
        InteractionType.RECOMMENDATION_WRITTEN: 10,
        InteractionType.RECOMMENDATION_RECEIVED: -10,
        InteractionType.ENDORSEMENT_GIVEN: 2,
        InteractionType.ENDORSEMENT_RECEIVED: -2,
        # Messages are neutral (bidirectional value exchange)
        InteractionType.MESSAGE_SENT: 0,
        InteractionType.MESSAGE_RECEIVED: 0,
        InteractionType.CONNECTION: 0,
    }

    DEFAULT_THRESHOLDS = {
        "strong_credit": 15,
        "credit": 5,
        "balanced_min": -5,
        "balanced_max": 5,
        "debit": -15,
    }

    def __init__(
        self,
        scores: Optional[dict[str, int]] = None,
        thresholds: Optional[dict[str, int]] = None,
    ):
        """Initialize ledger with configuration.

        Args:
            scores: Custom scores per interaction type
            thresholds: Custom thresholds for status classification
        """
        # Build scores dict
        self.scores = self.DEFAULT_SCORES.copy()
        if scores:
            for key, value in scores.items():
                if isinstance(key, str):
                    try:
                        int_type = InteractionType(key)
                        self.scores[int_type] = value
                    except ValueError:
                        logger.warning(f"Unknown interaction type: {key}")
                else:
                    self.scores[key] = value

        # Set thresholds
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)

    def _get_status(self, balance: float) -> ReciprocityStatus:
        """Determine reciprocity status from balance."""
        if balance >= self.thresholds["strong_credit"]:
            return ReciprocityStatus.STRONG_CREDIT
        elif balance >= self.thresholds["credit"]:
            return ReciprocityStatus.CREDIT
        elif balance >= self.thresholds["balanced_min"]:
            return ReciprocityStatus.BALANCED
        elif balance >= self.thresholds["debit"]:
            return ReciprocityStatus.DEBIT
        else:
            return ReciprocityStatus.STRONG_DEBIT

    def calculate_balance(
        self,
        relationship: Relationship,
    ) -> ReciprocityBalance:
        """Calculate reciprocity balance for a relationship.

        Args:
            relationship: The relationship to analyze

        Returns:
            ReciprocityBalance with breakdown
        """
        given: dict[str, int] = {}
        received: dict[str, int] = {}
        total_given = 0.0
        total_received = 0.0
        balance = 0.0

        for interaction in relationship.interactions:
            score = self.scores.get(interaction.type, 0)
            type_name = interaction.type.value

            if score > 0:
                # You gave value
                given[type_name] = given.get(type_name, 0) + 1
                total_given += score
                balance += score
            elif score < 0:
                # You received value
                received[type_name] = received.get(type_name, 0) + 1
                total_received += abs(score)
                balance += score

        status = self._get_status(balance)

        return ReciprocityBalance(
            person_id=relationship.person.id,
            person_name=relationship.person.display_name,
            balance=balance,
            status=status,
            given=given,
            received=received,
            total_given=total_given,
            total_received=total_received,
        )

    def calculate_network_balances(
        self,
        snapshot: NetworkSnapshot,
    ) -> tuple[NetworkSnapshot, list[ReciprocityBalance]]:
        """Calculate reciprocity balances for all relationships.

        Args:
            snapshot: Network snapshot to process

        Returns:
            Tuple of (updated snapshot, list of balances)
        """
        balances = []

        for person_id, relationship in snapshot.relationships.items():
            balance = self.calculate_balance(relationship)
            relationship.reciprocity_balance = balance.balance
            balances.append(balance)

        # Sort by balance (most credit first)
        balances.sort(key=lambda b: b.balance, reverse=True)

        logger.info(f"Calculated reciprocity balances for {len(balances)} relationships")

        return snapshot, balances

    def get_credit_relationships(
        self,
        snapshot: NetworkSnapshot,
        min_balance: float = 5.0,
    ) -> list[Relationship]:
        """Get relationships where you've given more value.

        Args:
            snapshot: Network snapshot
            min_balance: Minimum balance to include

        Returns:
            List of relationships sorted by balance (highest first)
        """
        credit_rels = [
            rel for rel in snapshot.relationships.values()
            if rel.reciprocity_balance >= min_balance
        ]
        return sorted(credit_rels, key=lambda r: r.reciprocity_balance, reverse=True)

    def get_debit_relationships(
        self,
        snapshot: NetworkSnapshot,
        max_balance: float = -5.0,
    ) -> list[Relationship]:
        """Get relationships where you've received more value.

        Args:
            snapshot: Network snapshot
            max_balance: Maximum balance to include (negative)

        Returns:
            List of relationships sorted by balance (lowest first)
        """
        debit_rels = [
            rel for rel in snapshot.relationships.values()
            if rel.reciprocity_balance <= max_balance
        ]
        return sorted(debit_rels, key=lambda r: r.reciprocity_balance)

    def get_summary(
        self,
        balances: list[ReciprocityBalance],
    ) -> dict:
        """Get summary statistics for reciprocity across network.

        Args:
            balances: List of reciprocity balances

        Returns:
            Summary statistics dictionary
        """
        if not balances:
            return {
                "total_relationships": 0,
                "by_status": {},
                "avg_balance": 0,
                "total_given": 0,
                "total_received": 0,
            }

        by_status = {}
        for status in ReciprocityStatus:
            count = sum(1 for b in balances if b.status == status)
            by_status[status.value] = count

        total_balance = sum(b.balance for b in balances)
        total_given = sum(b.total_given for b in balances)
        total_received = sum(b.total_received for b in balances)

        return {
            "total_relationships": len(balances),
            "by_status": by_status,
            "avg_balance": total_balance / len(balances),
            "total_given": total_given,
            "total_received": total_received,
            "net_balance": total_balance,
        }
