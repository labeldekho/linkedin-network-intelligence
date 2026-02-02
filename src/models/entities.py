"""
Core Data Models

Pydantic models representing LinkedIn network entities and their relationships.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class InteractionType(str, Enum):
    """Types of interactions tracked in the system."""
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    RECOMMENDATION_WRITTEN = "recommendation_written"
    RECOMMENDATION_RECEIVED = "recommendation_received"
    ENDORSEMENT_GIVEN = "endorsement_given"
    ENDORSEMENT_RECEIVED = "endorsement_received"
    CONNECTION = "connection"


class Person(BaseModel):
    """A person in the LinkedIn network."""
    id: str = Field(description="Unique identifier derived from name/profile")
    first_name: str
    last_name: str
    full_name: str = Field(default="")
    company: Optional[str] = None
    position: Optional[str] = None
    email: Optional[str] = None
    connected_on: Optional[datetime] = None
    profile_url: Optional[str] = None

    def model_post_init(self, __context) -> None:
        """Set full_name if not provided."""
        if not self.full_name:
            self.full_name = f"{self.first_name} {self.last_name}".strip()

    @property
    def display_name(self) -> str:
        """Human-readable name for display."""
        return self.full_name or f"{self.first_name} {self.last_name}"


class Interaction(BaseModel):
    """A single interaction with a person."""
    id: str = Field(default="", description="Unique interaction identifier")
    person_id: str = Field(description="ID of the person involved")
    type: InteractionType
    timestamp: datetime
    depth_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="LLM-assessed depth score (0-1)",
    )
    content_summary: Optional[str] = Field(
        default=None,
        description="Brief summary (never raw content)",
    )
    metadata: dict = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:
        """Generate ID if not provided."""
        if not self.id:
            self.id = f"{self.person_id}_{self.type.value}_{self.timestamp.isoformat()}"


class MessageThread(BaseModel):
    """A conversation thread with a person."""
    person_id: str
    messages: list[Interaction] = Field(default_factory=list)
    first_message: Optional[datetime] = None
    last_message: Optional[datetime] = None
    message_count: int = 0
    avg_depth_score: Optional[float] = None
    is_dormant: bool = False
    dormant_days: Optional[int] = None
    resurrection_score: Optional[float] = None
    resurrection_reason: Optional[str] = None

    def model_post_init(self, __context) -> None:
        """Calculate derived fields."""
        if self.messages:
            timestamps = [m.timestamp for m in self.messages]
            self.first_message = min(timestamps)
            self.last_message = max(timestamps)
            self.message_count = len(self.messages)
            depth_scores = [m.depth_score for m in self.messages if m.depth_score is not None]
            if depth_scores:
                self.avg_depth_score = sum(depth_scores) / len(depth_scores)


class Relationship(BaseModel):
    """A relationship with a person, including calculated metrics."""
    person: Person
    interactions: list[Interaction] = Field(default_factory=list)
    message_threads: list[MessageThread] = Field(default_factory=list)

    # Calculated scores
    strength_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Overall relationship strength (0+)",
    )
    reciprocity_balance: float = Field(
        default=0.0,
        description="Positive = you gave more, Negative = you received more",
    )

    # Derived metrics
    total_interactions: int = Field(default=0)
    first_interaction: Optional[datetime] = None
    last_interaction: Optional[datetime] = None
    days_since_last_interaction: Optional[int] = None
    is_dormant: bool = False

    # Enrichment data
    archetype: Optional[str] = Field(
        default=None,
        description="Relationship archetype classification",
    )
    warm_path_score: Optional[float] = None

    def model_post_init(self, __context) -> None:
        """Calculate derived fields."""
        if self.interactions:
            self.total_interactions = len(self.interactions)
            timestamps = [i.timestamp for i in self.interactions]
            self.first_interaction = min(timestamps)
            self.last_interaction = max(timestamps)
            self.days_since_last_interaction = (
                datetime.now() - self.last_interaction
            ).days

    @property
    def reciprocity_status(self) -> str:
        """Human-readable reciprocity status."""
        if self.reciprocity_balance >= 15:
            return "strong_credit"
        elif self.reciprocity_balance >= 5:
            return "credit"
        elif self.reciprocity_balance >= -5:
            return "balanced"
        elif self.reciprocity_balance >= -15:
            return "debit"
        else:
            return "strong_debit"


class NetworkSnapshot(BaseModel):
    """Container for an entire processed network."""
    generated_at: datetime = Field(default_factory=datetime.now)
    source_files: list[str] = Field(default_factory=list)

    # Core data
    people: dict[str, Person] = Field(default_factory=dict)
    relationships: dict[str, Relationship] = Field(default_factory=dict)

    # Summary statistics
    total_connections: int = 0
    total_interactions: int = 0
    avg_relationship_strength: float = 0.0
    network_archetype: Optional[str] = None

    # Enrichment flags
    is_enriched: bool = False
    enrichment_provider: Optional[str] = None

    def model_post_init(self, __context) -> None:
        """Calculate summary statistics."""
        self.total_connections = len(self.people)
        self.total_interactions = sum(
            len(r.interactions) for r in self.relationships.values()
        )
        if self.relationships:
            strengths = [r.strength_score for r in self.relationships.values()]
            self.avg_relationship_strength = sum(strengths) / len(strengths)

    def get_person(self, person_id: str) -> Optional[Person]:
        """Get a person by ID."""
        return self.people.get(person_id)

    def get_relationship(self, person_id: str) -> Optional[Relationship]:
        """Get a relationship by person ID."""
        return self.relationships.get(person_id)

    def get_top_relationships(self, n: int = 10) -> list[Relationship]:
        """Get top N relationships by strength score."""
        sorted_rels = sorted(
            self.relationships.values(),
            key=lambda r: r.strength_score,
            reverse=True,
        )
        return sorted_rels[:n]

    def get_dormant_relationships(
        self, min_days: int = 90, max_days: int = 1095
    ) -> list[Relationship]:
        """Get relationships that have been dormant for the specified period."""
        dormant = []
        for rel in self.relationships.values():
            if rel.days_since_last_interaction is not None:
                if min_days <= rel.days_since_last_interaction <= max_days:
                    dormant.append(rel)
        return sorted(
            dormant,
            key=lambda r: r.strength_score,
            reverse=True,
        )
