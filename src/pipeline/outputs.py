"""
Output Generation

Generates CSV, Markdown, and JSON reports from processed network data.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.entities import NetworkSnapshot, Relationship
from src.models.reciprocity import ReciprocityBalance, ReciprocityLedger
from src.models.warm_paths import WarmPathCandidate

logger = logging.getLogger(__name__)


class OutputGenerator:
    """Generates various output formats from network data."""

    def __init__(
        self,
        output_dir: str | Path = "./outputs",
        formats: Optional[list[str]] = None,
        timestamp_filenames: bool = True,
        max_items_per_section: int = 20,
        include_methodology: bool = True,
    ):
        """Initialize output generator.

        Args:
            output_dir: Directory for output files
            formats: List of formats to generate (csv, markdown, json)
            timestamp_filenames: Whether to include timestamp in filenames
            max_items_per_section: Maximum items per report section
            include_methodology: Whether to include methodology in reports
        """
        self.output_dir = Path(output_dir)
        self.formats = formats or ["csv", "markdown", "json"]
        self.timestamp_filenames = timestamp_filenames
        self.max_items_per_section = max_items_per_section
        self.include_methodology = include_methodology

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_filename(self, base_name: str, extension: str) -> Path:
        """Generate output filename."""
        if self.timestamp_filenames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{timestamp}.{extension}"
        else:
            filename = f"{base_name}.{extension}"
        return self.output_dir / filename

    def _relationships_to_csv(
        self,
        relationships: list[Relationship],
    ) -> str:
        """Convert relationships to CSV format."""
        lines = [
            "name,company,position,strength_score,reciprocity_balance,total_interactions,"
            "days_since_contact,is_dormant,archetype"
        ]

        for rel in relationships:
            lines.append(
                f'"{rel.person.display_name}",'
                f'"{rel.person.company or ""}",'
                f'"{rel.person.position or ""}",'
                f"{rel.strength_score:.3f},"
                f"{rel.reciprocity_balance:.1f},"
                f"{rel.total_interactions},"
                f"{rel.days_since_last_interaction or ''},"
                f"{rel.is_dormant},"
                f'"{rel.archetype or ""}"'
            )

        return "\n".join(lines)

    def _reciprocity_to_csv(
        self,
        balances: list[ReciprocityBalance],
    ) -> str:
        """Convert reciprocity balances to CSV format."""
        lines = [
            "name,balance,status,total_given,total_received,"
            "recs_written,recs_received,endorsements_given,endorsements_received"
        ]

        for b in balances:
            lines.append(
                f'"{b.person_name}",'
                f"{b.balance:.1f},"
                f'"{b.status.value}",'
                f"{b.total_given:.1f},"
                f"{b.total_received:.1f},"
                f'{b.given.get("recommendation_written", 0)},'
                f'{b.received.get("recommendation_received", 0)},'
                f'{b.given.get("endorsement_given", 0)},'
                f'{b.received.get("endorsement_received", 0)}'
            )

        return "\n".join(lines)

    def _warm_paths_to_csv(
        self,
        candidates: list[WarmPathCandidate],
    ) -> str:
        """Convert warm path candidates to CSV format."""
        lines = [
            "name,company,position,composite_score,warm_path_score,"
            "relationship_strength,connection_type,is_viable,days_since_contact"
        ]

        for c in candidates:
            lines.append(
                f'"{c.person_name}",'
                f'"{c.company or ""}",'
                f'"{c.position or ""}",'
                f"{c.composite_score:.3f},"
                f"{c.warm_path_score:.3f},"
                f"{c.relationship_strength:.3f},"
                f'"{c.connection_type}",'
                f"{c.is_viable_bridge},"
                f"{c.days_since_contact or ''}"
            )

        return "\n".join(lines)

    def _generate_relationship_strength_md(
        self,
        snapshot: NetworkSnapshot,
    ) -> str:
        """Generate relationship strength markdown report."""
        lines = ["# Relationship Strength Report\n"]

        if self.include_methodology:
            lines.extend([
                "## Methodology\n",
                "Relationship strength is calculated using a decay-based formula:\n",
                "- **Base weight**: Each interaction type has a base weight",
                "- **Depth multiplier**: LLM-assessed conversation depth (1.0-2.0x)",
                "- **Decay factor**: 0.5^(days_since / 180) - strength halves every 6 months\n",
            ])

        lines.extend([
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
            f"*Total connections: {snapshot.total_connections}*\n",
            f"*Average strength: {snapshot.avg_relationship_strength:.2f}*\n",
        ])

        # Top relationships
        top_rels = snapshot.get_top_relationships(self.max_items_per_section)

        lines.extend([
            "\n## Top Relationships\n",
            "| Rank | Name | Company | Strength | Last Contact |",
            "|------|------|---------|----------|--------------|",
        ])

        for i, rel in enumerate(top_rels, 1):
            last_contact = "N/A"
            if rel.days_since_last_interaction is not None:
                if rel.days_since_last_interaction == 0:
                    last_contact = "Today"
                elif rel.days_since_last_interaction == 1:
                    last_contact = "Yesterday"
                else:
                    last_contact = f"{rel.days_since_last_interaction}d ago"

            lines.append(
                f"| {i} | {rel.person.display_name} | "
                f"{rel.person.company or 'Unknown'} | "
                f"{rel.strength_score:.2f} | {last_contact} |"
            )

        return "\n".join(lines)

    def _generate_reciprocity_md(
        self,
        balances: list[ReciprocityBalance],
        summary: dict,
    ) -> str:
        """Generate reciprocity ledger markdown report."""
        lines = ["# Reciprocity Ledger\n"]

        if self.include_methodology:
            lines.extend([
                "## Methodology\n",
                "Tracks social capital balance in your network:\n",
                "- **Positive balance**: You've given more value (recommendations, endorsements)",
                "- **Negative balance**: You've received more value",
                "- **Balanced**: Healthy reciprocal relationship\n",
            ])

        lines.extend([
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
            "\n## Summary\n",
            f"- **Total relationships**: {summary.get('total_relationships', 0)}",
            f"- **Average balance**: {summary.get('avg_balance', 0):.1f}",
            f"- **Net social capital**: {summary.get('net_balance', 0):.1f}\n",
            "\n### Distribution\n",
        ])

        by_status = summary.get("by_status", {})
        for status, count in by_status.items():
            lines.append(f"- **{status}**: {count} relationships")

        # Credit relationships
        credit_rels = [b for b in balances if b.balance >= 5][:self.max_items_per_section]
        if credit_rels:
            lines.extend([
                "\n## Where You've Given More\n",
                "Consider: These people may be good candidates for asking favors.\n",
                "| Name | Balance | Status |",
                "|------|---------|--------|",
            ])
            for b in credit_rels:
                lines.append(f"| {b.person_name} | +{b.balance:.0f} | {b.status.value} |")

        # Debit relationships
        debit_rels = [b for b in balances if b.balance <= -5][:self.max_items_per_section]
        if debit_rels:
            lines.extend([
                "\n## Where You've Received More\n",
                "Consider: Look for opportunities to give back to these people.\n",
                "| Name | Balance | Status |",
                "|------|---------|--------|",
            ])
            for b in debit_rels:
                lines.append(f"| {b.person_name} | {b.balance:.0f} | {b.status.value} |")

        return "\n".join(lines)

    def _generate_resurrection_md(
        self,
        candidates: list[dict],
    ) -> str:
        """Generate resurrection candidates markdown report."""
        lines = ["# Resurrection Candidates\n"]

        lines.extend([
            "Dormant relationships worth rekindling.\n",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        ])

        if not candidates:
            lines.append("\n*No resurrection candidates found.*\n")
            return "\n".join(lines)

        worth_it = [c for c in candidates if c.get("should_resurrect", False)]
        lines.append(f"\n**{len(worth_it)} relationships recommended for outreach**\n")

        for i, c in enumerate(worth_it[:self.max_items_per_section], 1):
            person = c.get("person", {})
            lines.extend([
                f"\n## {i}. {person.get('name', 'Unknown')}\n",
                f"**Company**: {person.get('company', 'Unknown')}",
                f"**Position**: {person.get('position', 'Unknown')}",
                f"**Days dormant**: {c.get('days_dormant', 0)}",
                f"**Resurrection score**: {c.get('resurrection_score', 0):.2f}\n",
                f"### Why Reconnect\n{c.get('reasoning', 'N/A')}\n",
                f"### Suggested Approach\n{c.get('suggested_approach', 'N/A')}\n",
                f"### Conversation Starter\n> {c.get('conversation_starter', 'N/A')}\n",
            ])

        return "\n".join(lines)

    def _generate_warm_paths_md(
        self,
        candidates: list[WarmPathCandidate],
        target_company: str,
        summary: dict,
    ) -> str:
        """Generate warm paths markdown report."""
        lines = [f"# Warm Paths to {target_company}\n"]

        lines.extend([
            "Bridge candidates for warm introductions.\n",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
            "\n## Summary\n",
            f"- **Target company**: {target_company}",
            f"- **Candidates analyzed**: {summary.get('total_candidates', 0)}",
            f"- **Viable bridges**: {summary.get('viable_bridges', 0)}\n",
        ])

        if not candidates:
            lines.append("\n*No warm paths found. Consider expanding your network.*\n")
            return "\n".join(lines)

        viable = [c for c in candidates if c.is_viable_bridge]

        if viable:
            lines.append("\n## Best Paths\n")

            for i, c in enumerate(viable[:self.max_items_per_section], 1):
                lines.extend([
                    f"\n### {i}. {c.person_name}\n",
                    f"**Current company**: {c.company or 'Unknown'}",
                    f"**Position**: {c.position or 'Unknown'}",
                    f"**Connection type**: {c.connection_type}",
                    f"**Composite score**: {c.composite_score:.2f}",
                    f"**Your relationship strength**: {c.relationship_strength:.2f}\n",
                ])

                if c.reasoning:
                    lines.append(f"**Why this path**: {c.reasoning}\n")
                if c.approach_recommendation:
                    lines.append(f"**How to approach**: {c.approach_recommendation}\n")
                if c.ask_appropriateness:
                    lines.append(f"**Appropriateness**: {c.ask_appropriateness}\n")

        return "\n".join(lines)

    def _generate_network_summary_md(
        self,
        snapshot: NetworkSnapshot,
        archetype_analysis: dict,
        reciprocity_summary: dict,
    ) -> str:
        """Generate network summary markdown report."""
        lines = ["# Network Intelligence Summary\n"]

        lines.extend([
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
            "\n## Network Overview\n",
            f"- **Total connections**: {snapshot.total_connections}",
            f"- **Total interactions**: {snapshot.total_interactions}",
            f"- **Average relationship strength**: {snapshot.avg_relationship_strength:.2f}",
            f"- **Network archetype**: {snapshot.network_archetype or 'Not analyzed'}\n",
        ])

        # Archetype analysis
        if archetype_analysis:
            lines.extend([
                "\n## Network Archetype Analysis\n",
                f"**Primary archetype**: {archetype_analysis.get('primary_archetype', 'Unknown')}\n",
            ])

            if archetype_analysis.get("secondary_archetype"):
                lines.append(f"**Secondary archetype**: {archetype_analysis['secondary_archetype']}\n")

            health = archetype_analysis.get("network_health_score", 0)
            health_bar = "█" * int(health * 10) + "░" * (10 - int(health * 10))
            lines.append(f"**Network health**: [{health_bar}] {health:.0%}\n")

            if archetype_analysis.get("key_insight"):
                lines.append(f"**Key insight**: {archetype_analysis['key_insight']}\n")

            strengths = archetype_analysis.get("strengths", [])
            if strengths:
                lines.append("\n### Strengths\n")
                for s in strengths:
                    lines.append(f"- {s}")

            weaknesses = archetype_analysis.get("weaknesses", [])
            if weaknesses:
                lines.append("\n### Areas for Improvement\n")
                for w in weaknesses:
                    lines.append(f"- {w}")

            recommendations = archetype_analysis.get("recommendations", [])
            if recommendations:
                lines.append("\n### Recommendations\n")
                for r in recommendations:
                    lines.append(f"- {r}")

        # Reciprocity summary
        lines.extend([
            "\n## Reciprocity Overview\n",
            f"- **Average balance**: {reciprocity_summary.get('avg_balance', 0):.1f}",
            f"- **Net social capital**: {reciprocity_summary.get('net_balance', 0):.1f}\n",
        ])

        return "\n".join(lines)

    def generate_relationship_strength(
        self,
        snapshot: NetworkSnapshot,
    ) -> dict[str, Path]:
        """Generate relationship strength reports.

        Returns:
            Dictionary of format -> filepath
        """
        generated = {}

        # Sort relationships by strength
        sorted_rels = sorted(
            snapshot.relationships.values(),
            key=lambda r: r.strength_score,
            reverse=True,
        )

        if "csv" in self.formats:
            csv_content = self._relationships_to_csv(sorted_rels)
            filepath = self._get_filename("relationship_strength", "csv")
            filepath.write_text(csv_content)
            generated["csv"] = filepath

        if "markdown" in self.formats:
            md_content = self._generate_relationship_strength_md(snapshot)
            filepath = self._get_filename("relationship_strength", "md")
            filepath.write_text(md_content)
            generated["markdown"] = filepath

        if "json" in self.formats:
            json_data = [
                {
                    "name": r.person.display_name,
                    "company": r.person.company,
                    "position": r.person.position,
                    "strength_score": r.strength_score,
                    "reciprocity_balance": r.reciprocity_balance,
                    "total_interactions": r.total_interactions,
                    "days_since_contact": r.days_since_last_interaction,
                    "is_dormant": r.is_dormant,
                }
                for r in sorted_rels
            ]
            filepath = self._get_filename("relationship_strength", "json")
            filepath.write_text(json.dumps(json_data, indent=2))
            generated["json"] = filepath

        logger.info(f"Generated relationship strength reports: {list(generated.keys())}")
        return generated

    def generate_reciprocity_ledger(
        self,
        balances: list[ReciprocityBalance],
        summary: dict,
    ) -> dict[str, Path]:
        """Generate reciprocity ledger reports."""
        generated = {}

        if "csv" in self.formats:
            csv_content = self._reciprocity_to_csv(balances)
            filepath = self._get_filename("reciprocity_ledger", "csv")
            filepath.write_text(csv_content)
            generated["csv"] = filepath

        if "markdown" in self.formats:
            md_content = self._generate_reciprocity_md(balances, summary)
            filepath = self._get_filename("reciprocity_ledger", "md")
            filepath.write_text(md_content)
            generated["markdown"] = filepath

        if "json" in self.formats:
            json_data = {
                "summary": summary,
                "balances": [b.model_dump() for b in balances],
            }
            filepath = self._get_filename("reciprocity_ledger", "json")
            filepath.write_text(json.dumps(json_data, indent=2, default=str))
            generated["json"] = filepath

        logger.info(f"Generated reciprocity ledger reports: {list(generated.keys())}")
        return generated

    def generate_resurrection_candidates(
        self,
        candidates: list[dict],
    ) -> dict[str, Path]:
        """Generate resurrection candidates report."""
        generated = {}

        if "markdown" in self.formats:
            md_content = self._generate_resurrection_md(candidates)
            filepath = self._get_filename("resurrection_candidates", "md")
            filepath.write_text(md_content)
            generated["markdown"] = filepath

        if "json" in self.formats:
            filepath = self._get_filename("resurrection_candidates", "json")
            filepath.write_text(json.dumps(candidates, indent=2, default=str))
            generated["json"] = filepath

        logger.info(f"Generated resurrection reports: {list(generated.keys())}")
        return generated

    def generate_warm_paths(
        self,
        candidates: list[WarmPathCandidate],
        target_company: str,
        summary: dict,
    ) -> dict[str, Path]:
        """Generate warm paths reports."""
        generated = {}

        # Sanitize company name for filename
        safe_company = "".join(c if c.isalnum() else "_" for c in target_company.lower())

        if "csv" in self.formats:
            csv_content = self._warm_paths_to_csv(candidates)
            filepath = self._get_filename(f"warm_paths_{safe_company}", "csv")
            filepath.write_text(csv_content)
            generated["csv"] = filepath

        if "markdown" in self.formats:
            md_content = self._generate_warm_paths_md(candidates, target_company, summary)
            filepath = self._get_filename(f"warm_paths_{safe_company}", "md")
            filepath.write_text(md_content)
            generated["markdown"] = filepath

        if "json" in self.formats:
            json_data = {
                "target_company": target_company,
                "summary": summary,
                "candidates": [c.model_dump() for c in candidates],
            }
            filepath = self._get_filename(f"warm_paths_{safe_company}", "json")
            filepath.write_text(json.dumps(json_data, indent=2, default=str))
            generated["json"] = filepath

        logger.info(f"Generated warm paths reports for {target_company}: {list(generated.keys())}")
        return generated

    def generate_network_summary(
        self,
        snapshot: NetworkSnapshot,
        archetype_analysis: dict,
        reciprocity_summary: dict,
    ) -> dict[str, Path]:
        """Generate network summary report."""
        generated = {}

        if "markdown" in self.formats:
            md_content = self._generate_network_summary_md(
                snapshot, archetype_analysis, reciprocity_summary
            )
            filepath = self._get_filename("network_summary", "md")
            filepath.write_text(md_content)
            generated["markdown"] = filepath

        if "json" in self.formats:
            json_data = {
                "generated_at": datetime.now().isoformat(),
                "total_connections": snapshot.total_connections,
                "total_interactions": snapshot.total_interactions,
                "avg_relationship_strength": snapshot.avg_relationship_strength,
                "archetype": archetype_analysis,
                "reciprocity": reciprocity_summary,
            }
            filepath = self._get_filename("network_summary", "json")
            filepath.write_text(json.dumps(json_data, indent=2, default=str))
            generated["json"] = filepath

        logger.info(f"Generated network summary reports: {list(generated.keys())}")
        return generated


def generate_outputs(
    snapshot: NetworkSnapshot,
    analysis_results: dict,
    output_dir: str | Path = "./outputs",
    formats: Optional[list[str]] = None,
) -> dict[str, dict[str, Path]]:
    """Convenience function to generate all outputs.

    Args:
        snapshot: Processed network snapshot
        analysis_results: Results from enrichment pipeline
        output_dir: Output directory
        formats: Formats to generate

    Returns:
        Dictionary of report_type -> format -> filepath
    """
    generator = OutputGenerator(
        output_dir=output_dir,
        formats=formats or ["csv", "markdown", "json"],
    )

    results = {}

    # Relationship strength
    results["relationship_strength"] = generator.generate_relationship_strength(snapshot)

    # Reciprocity
    if "reciprocity_summary" in analysis_results:
        ledger = ReciprocityLedger()
        _, balances = ledger.calculate_network_balances(snapshot)
        results["reciprocity_ledger"] = generator.generate_reciprocity_ledger(
            balances,
            analysis_results["reciprocity_summary"],
        )

    # Resurrection candidates
    if "resurrection_candidates" in analysis_results:
        results["resurrection_candidates"] = generator.generate_resurrection_candidates(
            analysis_results["resurrection_candidates"]
        )

    # Network summary
    if "archetype" in analysis_results:
        results["network_summary"] = generator.generate_network_summary(
            snapshot,
            analysis_results.get("archetype", {}),
            analysis_results.get("reciprocity_summary", {}),
        )

    return results
