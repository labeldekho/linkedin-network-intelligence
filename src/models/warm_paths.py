"""
Warm Path Discovery

Identifies potential bridge candidates for introductions to target companies.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from src.llm.base import LLMProvider, get_provider
from src.models.entities import NetworkSnapshot, Relationship
from src.utils.cache import LLMCache

logger = logging.getLogger(__name__)


class WarmPathCandidate(BaseModel):
    """A potential bridge candidate for warm introduction."""
    person_id: str
    person_name: str
    company: Optional[str] = None
    position: Optional[str] = None

    # Scores
    relationship_strength: float = 0.0
    warm_path_score: float = Field(default=0.0, ge=0.0, le=1.0)
    composite_score: float = 0.0

    # Analysis results
    connection_type: str = "unknown"
    is_viable_bridge: bool = False
    reasoning: Optional[str] = None
    approach_recommendation: Optional[str] = None
    ask_appropriateness: Optional[str] = None

    # Metadata
    days_since_contact: Optional[int] = None
    last_interaction_type: Optional[str] = None


class WarmPathFinder:
    """Finds warm introduction paths to target companies."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        provider_name: str = "anthropic",
        model: Optional[str] = None,
        cache: Optional[LLMCache] = None,
        max_candidates: int = 10,
        min_bridge_strength: float = 0.3,
        weights: Optional[dict[str, float]] = None,
    ):
        """Initialize warm path finder.

        Args:
            provider: Pre-configured LLM provider
            provider_name: Provider name if provider not given
            model: Model name if provider not given
            cache: LLM response cache
            max_candidates: Maximum candidates to return
            min_bridge_strength: Minimum relationship strength for bridge
            weights: Scoring weights (relationship_strength, company_relevance, recency)
        """
        self.provider = provider or get_provider(provider_name, model=model)
        self.cache = cache or LLMCache(enabled=False)
        self.max_candidates = max_candidates
        self.min_bridge_strength = min_bridge_strength
        self.weights = weights or {
            "relationship_strength": 0.4,
            "company_relevance": 0.4,
            "recency": 0.2,
        }

        self._prompt_template: Optional[str] = None

    def _load_prompt(self) -> str:
        """Load warm path prompt template."""
        if self._prompt_template is None:
            from pathlib import Path
            prompt_file = Path(__file__).parent.parent.parent / "prompts" / "warm_path.txt"
            if prompt_file.exists():
                self._prompt_template = prompt_file.read_text()
            else:
                raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
        return self._prompt_template

    def _prefilter_candidates(
        self,
        snapshot: NetworkSnapshot,
        target_company: str,
    ) -> list[Relationship]:
        """Pre-filter relationships that might be relevant bridges.

        Uses simple heuristics before LLM analysis.
        """
        candidates = []
        target_lower = target_company.lower()

        for rel in snapshot.relationships.values():
            # Skip weak relationships
            if rel.strength_score < self.min_bridge_strength:
                continue

            # Prioritize current/former employees of target
            company = rel.person.company or ""
            if target_lower in company.lower():
                candidates.append(rel)
                continue

            # Include anyone with decent relationship strength
            # (LLM will assess industry/domain relevance)
            if rel.strength_score >= self.min_bridge_strength * 2:
                candidates.append(rel)

        # Sort by strength
        candidates.sort(key=lambda r: r.strength_score, reverse=True)

        # Limit for LLM analysis
        return candidates[:self.max_candidates * 2]

    async def _analyze_candidate(
        self,
        relationship: Relationship,
        target_company: str,
        target_role: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> dict:
        """Analyze a single candidate with LLM."""
        import json

        prompt_template = self._load_prompt()

        # Get last interaction info
        last_interaction = None
        last_interaction_type = None
        if relationship.interactions:
            sorted_interactions = sorted(
                relationship.interactions,
                key=lambda i: i.timestamp,
                reverse=True,
            )
            last = sorted_interactions[0]
            last_interaction = last.timestamp.strftime("%Y-%m-%d")
            last_interaction_type = last.type.value

        prompt = prompt_template.format(
            target_company=target_company,
            target_role=target_role or "Any",
            reason=reason or "Exploring opportunities",
            bridge_name=relationship.person.display_name,
            bridge_company=relationship.person.company or "Unknown",
            bridge_position=relationship.person.position or "Unknown",
            relationship_strength=round(relationship.strength_score * 10, 1),
            last_interaction=last_interaction or "Unknown",
            previous_companies="Unknown",  # Would need profile data
        )

        # Check cache
        cached = self.cache.get(
            prompt=prompt,
            system=None,
            model=self.provider.model,
            provider=self.provider.provider_name,
        )
        if cached:
            return json.loads(cached)

        try:
            result = await self.provider.complete_structured(
                prompt=prompt,
                schema={
                    "type": "object",
                    "properties": {
                        "warm_path_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "is_viable_bridge": {"type": "boolean"},
                        "connection_type": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "approach_recommendation": {"type": "string"},
                        "ask_appropriateness": {"type": "string"},
                        "alternative_value": {"type": "string"},
                    },
                    "required": ["warm_path_score", "is_viable_bridge", "connection_type"],
                },
            )

            # Cache result
            self.cache.set(
                prompt=prompt,
                system=None,
                model=self.provider.model,
                provider=self.provider.provider_name,
                content=json.dumps(result),
            )

            return result

        except Exception as e:
            logger.warning(f"Failed to analyze candidate {relationship.person.display_name}: {e}")
            return {
                "warm_path_score": 0.0,
                "is_viable_bridge": False,
                "connection_type": "unknown",
                "reasoning": "Analysis failed",
            }

    def _calculate_composite_score(
        self,
        relationship: Relationship,
        warm_path_score: float,
    ) -> float:
        """Calculate composite score combining multiple factors."""
        # Normalize relationship strength to 0-1 range (assuming max ~10)
        rel_score = min(relationship.strength_score / 10, 1.0)

        # Recency score (1.0 for recent, decays over time)
        recency_score = 0.5
        if relationship.days_since_last_interaction is not None:
            if relationship.days_since_last_interaction <= 30:
                recency_score = 1.0
            elif relationship.days_since_last_interaction <= 90:
                recency_score = 0.8
            elif relationship.days_since_last_interaction <= 180:
                recency_score = 0.6
            elif relationship.days_since_last_interaction <= 365:
                recency_score = 0.4
            else:
                recency_score = 0.2

        composite = (
            self.weights["relationship_strength"] * rel_score +
            self.weights["company_relevance"] * warm_path_score +
            self.weights["recency"] * recency_score
        )

        return composite

    async def find_warm_paths(
        self,
        snapshot: NetworkSnapshot,
        target_company: str,
        target_role: Optional[str] = None,
        reason: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> list[WarmPathCandidate]:
        """Find warm introduction paths to target company.

        Args:
            snapshot: Network snapshot to search
            target_company: Target company name
            target_role: Optional target role/department
            reason: Reason for seeking introduction
            progress_callback: Optional progress callback

        Returns:
            List of WarmPathCandidate sorted by score
        """
        # Pre-filter candidates
        relationships = self._prefilter_candidates(snapshot, target_company)

        if not relationships:
            logger.info(f"No potential bridges found for {target_company}")
            return []

        logger.info(f"Analyzing {len(relationships)} potential bridges to {target_company}")

        # Analyze candidates with LLM
        results = []
        batch_size = 5

        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]

            tasks = [
                self._analyze_candidate(rel, target_company, target_role, reason)
                for rel in batch
            ]

            analyses = await asyncio.gather(*tasks, return_exceptions=True)

            for rel, analysis in zip(batch, analyses):
                if isinstance(analysis, Exception):
                    logger.warning(f"Analysis failed for {rel.person.display_name}: {analysis}")
                    continue

                warm_path_score = analysis.get("warm_path_score", 0.0)
                composite_score = self._calculate_composite_score(rel, warm_path_score)

                # Get last interaction info
                last_interaction_type = None
                if rel.interactions:
                    sorted_interactions = sorted(
                        rel.interactions,
                        key=lambda i: i.timestamp,
                        reverse=True,
                    )
                    last_interaction_type = sorted_interactions[0].type.value

                candidate = WarmPathCandidate(
                    person_id=rel.person.id,
                    person_name=rel.person.display_name,
                    company=rel.person.company,
                    position=rel.person.position,
                    relationship_strength=rel.strength_score,
                    warm_path_score=warm_path_score,
                    composite_score=composite_score,
                    connection_type=analysis.get("connection_type", "unknown"),
                    is_viable_bridge=analysis.get("is_viable_bridge", False),
                    reasoning=analysis.get("reasoning"),
                    approach_recommendation=analysis.get("approach_recommendation"),
                    ask_appropriateness=analysis.get("ask_appropriateness"),
                    days_since_contact=rel.days_since_last_interaction,
                    last_interaction_type=last_interaction_type,
                )
                results.append(candidate)

            if progress_callback:
                progress_callback(min(i + batch_size, len(relationships)), len(relationships))

        # Sort by composite score and filter
        results.sort(key=lambda c: c.composite_score, reverse=True)

        # Return top candidates that are viable
        viable = [c for c in results if c.is_viable_bridge][:self.max_candidates]

        # If not enough viable, include top non-viable as alternatives
        if len(viable) < self.max_candidates:
            non_viable = [c for c in results if not c.is_viable_bridge]
            viable.extend(non_viable[:self.max_candidates - len(viable)])

        logger.info(f"Found {len(viable)} warm path candidates for {target_company}")

        return viable

    def get_summary(
        self,
        candidates: list[WarmPathCandidate],
        target_company: str,
    ) -> dict:
        """Get summary of warm path analysis.

        Args:
            candidates: List of candidates
            target_company: Target company name

        Returns:
            Summary dictionary
        """
        if not candidates:
            return {
                "target_company": target_company,
                "total_candidates": 0,
                "viable_bridges": 0,
                "connection_types": {},
                "best_candidate": None,
            }

        viable = [c for c in candidates if c.is_viable_bridge]

        connection_types = {}
        for c in candidates:
            ct = c.connection_type
            connection_types[ct] = connection_types.get(ct, 0) + 1

        best = candidates[0] if candidates else None

        return {
            "target_company": target_company,
            "total_candidates": len(candidates),
            "viable_bridges": len(viable),
            "connection_types": connection_types,
            "best_candidate": {
                "name": best.person_name,
                "company": best.company,
                "score": best.composite_score,
                "connection_type": best.connection_type,
            } if best else None,
            "avg_relationship_strength": (
                sum(c.relationship_strength for c in candidates) / len(candidates)
                if candidates else 0
            ),
        }
