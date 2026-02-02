"""
Entity Resolution and Normalization

Resolves entities across multiple data sources and builds unified timeline.
"""

import hashlib
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from src.models.entities import (
    Interaction,
    InteractionType,
    MessageThread,
    NetworkSnapshot,
    Person,
    Relationship,
)
from src.pipeline.ingest import LinkedInExport

logger = logging.getLogger(__name__)


class PersonResolver:
    """Resolves and deduplicates person entities across data sources."""

    def __init__(self):
        self._people: dict[str, Person] = {}
        self._name_to_id: dict[str, str] = {}
        self._email_to_id: dict[str, str] = {}

    def _generate_id(self, first_name: str, last_name: str, email: Optional[str] = None) -> str:
        """Generate a deterministic person ID."""
        # Normalize names
        first = first_name.lower().strip()
        last = last_name.lower().strip()

        # Use email if available for more unique ID
        if email:
            key = f"{email}".lower()
        else:
            key = f"{first}:{last}"

        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for matching."""
        # Remove special characters, lowercase, collapse spaces
        name = re.sub(r"[^\w\s]", "", name.lower())
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _create_name_key(self, first_name: str, last_name: str) -> str:
        """Create a lookup key from names."""
        return f"{self._normalize_name(first_name)}:{self._normalize_name(last_name)}"

    def add_person(
        self,
        first_name: str,
        last_name: str,
        email: Optional[str] = None,
        company: Optional[str] = None,
        position: Optional[str] = None,
        connected_on: Optional[datetime] = None,
        profile_url: Optional[str] = None,
    ) -> Person:
        """Add or update a person in the resolver.

        Returns existing person if already known, otherwise creates new.
        """
        # Check by email first
        if email and email.lower() in self._email_to_id:
            person_id = self._email_to_id[email.lower()]
            person = self._people[person_id]
            # Update fields if they were empty
            if not person.company and company:
                person.company = company
            if not person.position and position:
                person.position = position
            if not person.connected_on and connected_on:
                person.connected_on = connected_on
            return person

        # Check by name
        name_key = self._create_name_key(first_name, last_name)
        if name_key in self._name_to_id:
            person_id = self._name_to_id[name_key]
            person = self._people[person_id]
            # Update fields
            if email and not person.email:
                person.email = email
                self._email_to_id[email.lower()] = person_id
            if not person.company and company:
                person.company = company
            if not person.position and position:
                person.position = position
            if not person.connected_on and connected_on:
                person.connected_on = connected_on
            return person

        # Create new person
        person_id = self._generate_id(first_name, last_name, email)

        # Handle ID collision
        if person_id in self._people:
            # Append suffix to make unique
            suffix = 1
            while f"{person_id}_{suffix}" in self._people:
                suffix += 1
            person_id = f"{person_id}_{suffix}"

        person = Person(
            id=person_id,
            first_name=first_name,
            last_name=last_name,
            email=email,
            company=company,
            position=position,
            connected_on=connected_on,
            profile_url=profile_url,
        )

        self._people[person_id] = person
        self._name_to_id[name_key] = person_id
        if email:
            self._email_to_id[email.lower()] = person_id

        return person

    def resolve_by_name(self, full_name: str) -> Optional[Person]:
        """Try to resolve a person by full name."""
        if not full_name:
            return None

        # Split full name into first and last
        parts = full_name.strip().split()
        if len(parts) == 0:
            return None
        elif len(parts) == 1:
            first_name = parts[0]
            last_name = ""
        else:
            first_name = parts[0]
            last_name = " ".join(parts[1:])

        name_key = self._create_name_key(first_name, last_name)
        if name_key in self._name_to_id:
            return self._people[self._name_to_id[name_key]]

        # Try reverse (last name first)
        name_key_rev = self._create_name_key(last_name, first_name)
        if name_key_rev in self._name_to_id:
            return self._people[self._name_to_id[name_key_rev]]

        return None

    def get_all_people(self) -> dict[str, Person]:
        """Return all resolved people."""
        return self._people.copy()


def _build_message_threads(
    interactions: list[Interaction],
    thread_break_days: int = 7,
) -> list[MessageThread]:
    """Group message interactions into conversation threads."""
    # Filter to message interactions only
    messages = [
        i for i in interactions
        if i.type in (InteractionType.MESSAGE_SENT, InteractionType.MESSAGE_RECEIVED)
    ]

    if not messages:
        return []

    # Sort by timestamp
    messages.sort(key=lambda x: x.timestamp)

    threads = []
    current_thread_messages = [messages[0]]

    for msg in messages[1:]:
        prev_msg = current_thread_messages[-1]
        gap = (msg.timestamp - prev_msg.timestamp).days

        if gap > thread_break_days:
            # Start new thread
            threads.append(MessageThread(
                person_id=current_thread_messages[0].person_id,
                messages=current_thread_messages,
            ))
            current_thread_messages = [msg]
        else:
            current_thread_messages.append(msg)

    # Don't forget the last thread
    if current_thread_messages:
        threads.append(MessageThread(
            person_id=current_thread_messages[0].person_id,
            messages=current_thread_messages,
        ))

    return threads


def normalize_network(
    export: LinkedInExport,
    thread_break_days: int = 7,
    owner_name: Optional[str] = None,
) -> NetworkSnapshot:
    """Normalize LinkedIn export into unified network snapshot.

    Args:
        export: Loaded LinkedIn export data
        thread_break_days: Days gap to consider new conversation thread
        owner_name: Your name (to filter out from messages)

    Returns:
        Normalized NetworkSnapshot with all entities resolved
    """
    resolver = PersonResolver()
    interactions_by_person: dict[str, list[Interaction]] = {}

    # Process connections first
    for conn in export.connections:
        person = resolver.add_person(
            first_name=conn.first_name,
            last_name=conn.last_name,
            email=conn.email_address,
            company=conn.company,
            position=conn.position,
            connected_on=conn.connected_on,
            profile_url=conn.url,
        )

        # Add connection interaction
        if conn.connected_on:
            interaction = Interaction(
                person_id=person.id,
                type=InteractionType.CONNECTION,
                timestamp=conn.connected_on,
            )
            if person.id not in interactions_by_person:
                interactions_by_person[person.id] = []
            interactions_by_person[person.id].append(interaction)

    # Process messages
    owner_name_lower = owner_name.lower() if owner_name else None

    for msg in export.messages:
        # Determine if we sent or received
        from_name = msg.from_name
        to_name = msg.to_name

        # Skip if we can't identify the other party
        if not from_name and not to_name:
            continue

        # Determine the other person and interaction type
        if owner_name_lower:
            if from_name and from_name.lower() == owner_name_lower:
                # We sent this message
                other_name = to_name
                interaction_type = InteractionType.MESSAGE_SENT
            else:
                # We received this message
                other_name = from_name
                interaction_type = InteractionType.MESSAGE_RECEIVED
        else:
            # Without owner name, use conversation context
            # Assume the "from" person is the other party for received messages
            other_name = from_name
            interaction_type = InteractionType.MESSAGE_RECEIVED

        if not other_name:
            continue

        person = resolver.resolve_by_name(other_name)
        if not person:
            # Create new person if not in connections
            parts = other_name.strip().split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = " ".join(parts[1:])
            else:
                first_name = other_name
                last_name = ""

            person = resolver.add_person(
                first_name=first_name,
                last_name=last_name,
            )

        interaction = Interaction(
            person_id=person.id,
            type=interaction_type,
            timestamp=msg.date,
            content_summary=msg.subject,  # Only store subject, not content
            metadata={"conversation_id": msg.conversation_id},
        )

        if person.id not in interactions_by_person:
            interactions_by_person[person.id] = []
        interactions_by_person[person.id].append(interaction)

    # Process endorsements given
    for endorsement in export.endorsements_given:
        if not endorsement.endorsee_name:
            continue

        person = resolver.resolve_by_name(endorsement.endorsee_name)
        if not person:
            parts = endorsement.endorsee_name.strip().split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = " ".join(parts[1:])
            else:
                first_name = endorsement.endorsee_name
                last_name = ""
            person = resolver.add_person(first_name=first_name, last_name=last_name)

        interaction = Interaction(
            person_id=person.id,
            type=InteractionType.ENDORSEMENT_GIVEN,
            timestamp=endorsement.date or datetime.now(),
            metadata={"skill": endorsement.skill},
        )

        if person.id not in interactions_by_person:
            interactions_by_person[person.id] = []
        interactions_by_person[person.id].append(interaction)

    # Process endorsements received
    for endorsement in export.endorsements_received:
        if not endorsement.endorser_name:
            continue

        person = resolver.resolve_by_name(endorsement.endorser_name)
        if not person:
            parts = endorsement.endorser_name.strip().split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = " ".join(parts[1:])
            else:
                first_name = endorsement.endorser_name
                last_name = ""
            person = resolver.add_person(first_name=first_name, last_name=last_name)

        interaction = Interaction(
            person_id=person.id,
            type=InteractionType.ENDORSEMENT_RECEIVED,
            timestamp=endorsement.date or datetime.now(),
            metadata={"skill": endorsement.skill},
        )

        if person.id not in interactions_by_person:
            interactions_by_person[person.id] = []
        interactions_by_person[person.id].append(interaction)

    # Process recommendations given
    for rec in export.recommendations_given:
        full_name = f"{rec.first_name or ''} {rec.last_name or ''}".strip()
        if not full_name:
            continue

        person = resolver.resolve_by_name(full_name)
        if not person:
            person = resolver.add_person(
                first_name=rec.first_name or "",
                last_name=rec.last_name or "",
                company=rec.company,
                position=rec.job_title,
            )

        interaction = Interaction(
            person_id=person.id,
            type=InteractionType.RECOMMENDATION_WRITTEN,
            timestamp=rec.date or datetime.now(),
        )

        if person.id not in interactions_by_person:
            interactions_by_person[person.id] = []
        interactions_by_person[person.id].append(interaction)

    # Process recommendations received
    for rec in export.recommendations_received:
        full_name = f"{rec.first_name or ''} {rec.last_name or ''}".strip()
        if not full_name:
            continue

        person = resolver.resolve_by_name(full_name)
        if not person:
            person = resolver.add_person(
                first_name=rec.first_name or "",
                last_name=rec.last_name or "",
                company=rec.company,
                position=rec.job_title,
            )

        interaction = Interaction(
            person_id=person.id,
            type=InteractionType.RECOMMENDATION_RECEIVED,
            timestamp=rec.date or datetime.now(),
        )

        if person.id not in interactions_by_person:
            interactions_by_person[person.id] = []
        interactions_by_person[person.id].append(interaction)

    # Build relationships
    people = resolver.get_all_people()
    relationships: dict[str, Relationship] = {}

    for person_id, person in people.items():
        interactions = interactions_by_person.get(person_id, [])

        # Build message threads
        threads = _build_message_threads(interactions, thread_break_days)

        relationship = Relationship(
            person=person,
            interactions=interactions,
            message_threads=threads,
        )

        relationships[person_id] = relationship

    # Create snapshot
    snapshot = NetworkSnapshot(
        source_files=export.loaded_files,
        people=people,
        relationships=relationships,
    )

    logger.info(
        f"Network normalized: {len(people)} people, "
        f"{sum(len(r.interactions) for r in relationships.values())} interactions, "
        f"{sum(len(r.message_threads) for r in relationships.values())} message threads"
    )

    return snapshot
