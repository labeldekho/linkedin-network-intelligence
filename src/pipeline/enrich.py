"""
LLM Enrichment Pipeline

Orchestrates LLM-based analysis for message depth, resurrection, and network insights.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.llm.base import LLMProvider, get_provider
from src.models.entities import (
    InteractionType,
    MessageThread,
    NetworkSnapshot,
    Relationship,
)
from src.utils.cache import LLMCache

logger = logging.getLogger(__name__)


class EnrichmentPipeline:
    """Orchestrates LLM-based enrichment of network data."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        provider_name: str = "anthropic",
        model: Optional[str] = None,
        cache: Optional[LLMCache] = None,
        prompts_dir: Optional[Path] = None,
        batch_size: int = 5,
        min_thread_length: int = 2,
    ):
        """Initialize enrichment pipeline.

        Args:
            provider: Pre-configured LLM provider
            provider_name: Provider name if provider not given
            model: Model name if provider not given
            cache: LLM response cache
            prompts_dir: Directory containing prompt templates
            batch_size: Number of concurrent LLM calls
            min_thread_length: Minimum messages to analyze thread
        """
        self.provider = provider or get_provider(provider_name, model=model)
        self.cache = cache or LLMCache(enabled=False)
        self.batch_size = batch_size
        self.min_thread_length = min_thread_length

        # Load prompt templates
        self.prompts_dir = prompts_dir or Path(__file__).parent.parent.parent / "prompts"
        self._prompts: dict[str, str] = {}

    def _load_prompt(self, name: str) -> str:
        """Load a prompt template by name."""
        if name not in self._prompts:
            prompt_file = self.prompts_dir / f"{name}.txt"
            if prompt_file.exists():
                self._prompts[name] = prompt_file.read_text()
            else:
                raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
        return self._prompts[name]

    async def _analyze_message_thread(
        self,
        thread: MessageThread,
        person_name: str,
    ) -> dict:
        """Analyze a single message thread for depth."""
        prompt_template = self._load_prompt("message_depth")

        # Build conversation summary (without actual content)
        message_types = [
            "sent" if m.type == InteractionType.MESSAGE_SENT else "received"
            for m in thread.messages
        ]

        time_span = "N/A"
        if thread.first_message and thread.last_message:
            days = (thread.last_message - thread.first_message).days
            time_span = f"{days} days"

        prompt = prompt_template.format(
            participants=f"You and {person_name}",
            message_count=thread.message_count,
            time_span=time_span,
            conversation_summary=f"Exchange of {len(message_types)} messages: {', '.join(message_types[:10])}{'...' if len(message_types) > 10 else ''}",
        )

        # Check cache
        cached = self.cache.get(
            prompt=prompt,
            system=None,
            model=self.provider.model,
            provider=self.provider.provider_name,
        )
        if cached:
            import json
            return json.loads(cached)

        # Call LLM
        try:
            result = await self.provider.complete_structured(
                prompt=prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "depth_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "reasoning": {"type": "string"},
                        "key_topics": {"type": "array", "items": {"type": "string"}},
                        "relationship_signals": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["depth_score", "reasoning"],
                },
            )

            # Cache result
            import json
            self.cache.set(
                prompt=prompt,
                system=None,
                model=self.provider.model,
                provider=self.provider.provider_name,
                content=json.dumps(result),
            )

            return result

        except Exception as e:
            logger.warning(f"Failed to analyze message thread: {e}")
            return {"depth_score": 0.5, "reasoning": "Analysis failed"}

    async def _analyze_resurrection_candidate(
        self,
        relationship: Relationship,
        user_context: Optional[dict] = None,
    ) -> dict:
        """Analyze a dormant relationship for resurrection potential."""
        prompt_template = self._load_prompt("resurrection")
        user_context = user_context or {}

        # Summarize past interactions
        interaction_types = {}
        for interaction in relationship.interactions:
            type_name = interaction.type.value
            interaction_types[type_name] = interaction_types.get(type_name, 0) + 1

        interaction_summary = ", ".join(
            f"{count} {type_name}" for type_name, count in interaction_types.items()
        )

        prompt = prompt_template.format(
            person_name=relationship.person.display_name,
            company=relationship.person.company or "Unknown",
            position=relationship.person.position or "Unknown",
            days_dormant=relationship.days_since_last_interaction or 0,
            relationship_strength=round(relationship.strength_score * 10, 1),
            interaction_summary=interaction_summary or "Initial connection only",
            user_role=user_context.get("role", "Not specified"),
            user_company=user_context.get("company", "Not specified"),
            user_focus=user_context.get("focus", "Not specified"),
        )

        # Check cache
        cached = self.cache.get(
            prompt=prompt,
            system=None,
            model=self.provider.model,
            provider=self.provider.provider_name,
        )
        if cached:
            import json
            return json.loads(cached)

        try:
            result = await self.provider.complete_structured(
                prompt=prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "resurrection_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "should_resurrect": {"type": "boolean"},
                        "reasoning": {"type": "string"},
                        "suggested_approach": {"type": "string"},
                        "conversation_starter": {"type": "string"},
                        "value_you_can_offer": {"type": "string"},
                        "potential_value_to_you": {"type": "string"},
                        "timing_assessment": {"type": "string"},
                    },
                    "required": ["resurrection_score", "should_resurrect", "reasoning"],
                },
            )

            import json
            self.cache.set(
                prompt=prompt,
                system=None,
                model=self.provider.model,
                provider=self.provider.provider_name,
                content=json.dumps(result),
            )

            return result

        except Exception as e:
            logger.warning(f"Failed to analyze resurrection candidate: {e}")
            return {"resurrection_score": 0.0, "should_resurrect": False, "reasoning": "Analysis failed"}

    async def _analyze_network_archetype(
        self,
        snapshot: NetworkSnapshot,
        reciprocity_summary: dict,
    ) -> dict:
        """Analyze network to determine its archetype."""
        prompt_template = self._load_prompt("archetype")

        # Calculate statistics
        now = datetime.now()
        active_count = sum(
            1 for r in snapshot.relationships.values()
            if r.last_interaction and (now - r.last_interaction).days <= 90
        )
        dormant_count = len(snapshot.relationships) - active_count

        # Company distribution
        companies: dict[str, int] = {}
        for person in snapshot.people.values():
            if person.company:
                companies[person.company] = companies.get(person.company, 0) + 1

        top_companies = sorted(companies.items(), key=lambda x: x[1], reverse=True)[:10]
        company_dist = "\n".join(
            f"- {company}: {count} connections" for company, count in top_companies
        )

        # Interaction counts
        interaction_counts = {t.value: 0 for t in InteractionType}
        for rel in snapshot.relationships.values():
            for interaction in rel.interactions:
                interaction_counts[interaction.type.value] += 1

        prompt = prompt_template.format(
            total_connections=snapshot.total_connections,
            active_count=active_count,
            dormant_count=dormant_count,
            avg_strength=round(snapshot.avg_relationship_strength, 2),
            company_distribution=company_dist or "No company data available",
            messages_sent=interaction_counts.get("message_sent", 0),
            messages_received=interaction_counts.get("message_received", 0),
            recs_given=interaction_counts.get("recommendation_written", 0),
            recs_received=interaction_counts.get("recommendation_received", 0),
            endorsements_given=interaction_counts.get("endorsement_given", 0),
            endorsements_received=interaction_counts.get("endorsement_received", 0),
            strong_credit_count=reciprocity_summary.get("by_status", {}).get("strong_credit", 0),
            balanced_count=reciprocity_summary.get("by_status", {}).get("balanced", 0),
            strong_debit_count=reciprocity_summary.get("by_status", {}).get("strong_debit", 0),
        )

        try:
            result = await self.provider.complete_structured(
                prompt=prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "primary_archetype": {"type": "string"},
                        "secondary_archetype": {"type": ["string", "null"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "network_health_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "strengths": {"type": "array", "items": {"type": "string"}},
                        "weaknesses": {"type": "array", "items": {"type": "string"}},
                        "recommendations": {"type": "array", "items": {"type": "string"}},
                        "key_insight": {"type": "string"},
                    },
                    "required": ["primary_archetype", "network_health_score", "key_insight"],
                },
            )
            return result

        except Exception as e:
            logger.warning(f"Failed to analyze network archetype: {e}")
            return {
                "primary_archetype": "Unknown",
                "network_health_score": 0.5,
                "key_insight": "Analysis failed",
            }

    async def enrich_message_depths(
        self,
        snapshot: NetworkSnapshot,
        progress_callback: Optional[callable] = None,
    ) -> NetworkSnapshot:
        """Enrich message threads with depth scores.

        Args:
            snapshot: Network snapshot to enrich
            progress_callback: Optional callback for progress updates

        Returns:
            Enriched NetworkSnapshot
        """
        # Collect threads to analyze
        threads_to_analyze = []
        for rel in snapshot.relationships.values():
            for thread in rel.message_threads:
                if thread.message_count >= self.min_thread_length:
                    threads_to_analyze.append((rel.person.display_name, thread))

        if not threads_to_analyze:
            logger.info("No message threads to analyze")
            return snapshot

        logger.info(f"Analyzing {len(threads_to_analyze)} message threads")

        # Process in batches
        for i in range(0, len(threads_to_analyze), self.batch_size):
            batch = threads_to_analyze[i:i + self.batch_size]

            tasks = [
                self._analyze_message_thread(thread, person_name)
                for person_name, thread in batch
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (person_name, thread), result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to analyze thread for {person_name}: {result}")
                    continue

                thread.avg_depth_score = result.get("depth_score", 0.5)

                # Update individual message depth scores
                for msg in thread.messages:
                    msg.depth_score = thread.avg_depth_score

            if progress_callback:
                progress_callback(min(i + self.batch_size, len(threads_to_analyze)), len(threads_to_analyze))

        snapshot.is_enriched = True
        snapshot.enrichment_provider = self.provider.provider_name

        return snapshot

    async def find_resurrection_candidates(
        self,
        snapshot: NetworkSnapshot,
        min_dormant_days: int = 90,
        max_dormant_days: int = 1095,
        min_strength_threshold: float = 0.2,
        user_context: Optional[dict] = None,
        max_candidates: int = 20,
    ) -> list[dict]:
        """Find dormant relationships worth rekindling.

        Args:
            snapshot: Network snapshot
            min_dormant_days: Minimum days since last contact
            max_dormant_days: Maximum days (too old to resurrect)
            min_strength_threshold: Minimum relationship strength
            user_context: Dict with user's role, company, focus
            max_candidates: Maximum candidates to return

        Returns:
            List of resurrection candidate analyses
        """
        # Filter candidates
        candidates = []
        for rel in snapshot.relationships.values():
            if rel.days_since_last_interaction is None:
                continue
            if not (min_dormant_days <= rel.days_since_last_interaction <= max_dormant_days):
                continue
            if rel.strength_score < min_strength_threshold:
                continue
            candidates.append(rel)

        # Sort by strength and take top candidates
        candidates.sort(key=lambda r: r.strength_score, reverse=True)
        candidates = candidates[:max_candidates]

        if not candidates:
            logger.info("No resurrection candidates found")
            return []

        logger.info(f"Analyzing {len(candidates)} resurrection candidates")

        # Analyze candidates
        results = []
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]

            tasks = [
                self._analyze_resurrection_candidate(rel, user_context)
                for rel in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for rel, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to analyze {rel.person.display_name}: {result}")
                    continue

                result["person"] = {
                    "id": rel.person.id,
                    "name": rel.person.display_name,
                    "company": rel.person.company,
                    "position": rel.person.position,
                }
                result["relationship_strength"] = rel.strength_score
                result["days_dormant"] = rel.days_since_last_interaction
                results.append(result)

        # Sort by resurrection score
        results.sort(key=lambda r: r.get("resurrection_score", 0), reverse=True)

        return results

    async def analyze_network(
        self,
        snapshot: NetworkSnapshot,
        reciprocity_summary: dict,
    ) -> dict:
        """Analyze network and determine its archetype.

        Args:
            snapshot: Network snapshot
            reciprocity_summary: Summary from ReciprocityLedger

        Returns:
            Network analysis results
        """
        result = await self._analyze_network_archetype(snapshot, reciprocity_summary)
        snapshot.network_archetype = result.get("primary_archetype")
        return result


async def enrich_network(
    snapshot: NetworkSnapshot,
    provider_name: str = "anthropic",
    model: Optional[str] = None,
    cache_enabled: bool = True,
    user_context: Optional[dict] = None,
) -> tuple[NetworkSnapshot, dict]:
    """Convenience function to fully enrich a network snapshot.

    Args:
        snapshot: Network snapshot to enrich
        provider_name: LLM provider to use
        model: Model name
        cache_enabled: Whether to use response cache
        user_context: User's context for resurrection analysis

    Returns:
        Tuple of (enriched snapshot, analysis results)
    """
    from src.models.reciprocity import ReciprocityLedger

    cache = LLMCache(enabled=cache_enabled)
    pipeline = EnrichmentPipeline(
        provider_name=provider_name,
        model=model,
        cache=cache,
    )

    # Enrich message depths
    snapshot = await pipeline.enrich_message_depths(snapshot)

    # Calculate reciprocity
    ledger = ReciprocityLedger()
    snapshot, balances = ledger.calculate_network_balances(snapshot)
    reciprocity_summary = ledger.get_summary(balances)

    # Analyze network archetype
    archetype_analysis = await pipeline.analyze_network(snapshot, reciprocity_summary)

    # Find resurrection candidates
    resurrection_candidates = await pipeline.find_resurrection_candidates(
        snapshot,
        user_context=user_context,
    )

    results = {
        "archetype": archetype_analysis,
        "resurrection_candidates": resurrection_candidates,
        "reciprocity_summary": reciprocity_summary,
    }

    return snapshot, results
