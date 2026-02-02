# LinkedIn Network Intelligence

Transform your LinkedIn data export into actionable relationship intelligence.

**This is not a LinkedIn clone.** It's an analysis layer that LinkedIn intentionally doesn't provide.

---

## What This Does

- **Relationship Strength Scoring** — Know who your actual connections are vs. noise
- **Reciprocity Tracking** — See where you've given vs. received value
- **Conversation Resurrection** — Find dormant threads worth reviving
- **Warm Path Discovery** — Find bridges to target companies through your network
- **Network Archetype Analysis** — Understand your networking style and blind spots

---

## Privacy First

- Uses **only your exported data** — no scraping, no LinkedIn API
- Runs **locally** — your data never leaves your machine (unless using cloud LLM)
- Supports **on-premise LLMs** — Llama, Qwen via Ollama for full data sovereignty
- **No telemetry** — we don't track usage or collect data

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourname/linkedin-network-intelligence.git
cd linkedin-network-intelligence
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set API key (or use local model)
export ANTHROPIC_API_KEY=your-key-here

# Export your LinkedIn data (see instructions below)
# Place files in ./data/

# Run analysis
python -m src.main process --input ./data/ --output ./outputs/
```

---

## Getting Your LinkedIn Data

1. Go to LinkedIn → Settings → Data Privacy → Get a copy of your data
2. Select "Download larger data archive"
3. Wait for email (can take 24-72 hours)
4. Download and extract the ZIP
5. Copy relevant CSV files to `./data/`

**Required files:**
- `Connections.csv`
- `Messages.csv`

**Optional but recommended:**
- `Endorsements.csv`
- `Recommendations_Given.csv`
- `Recommendations_Received.csv`
- `Profile.csv`

---

## Using Different LLM Providers

### Claude (Default)
```bash
export ANTHROPIC_API_KEY=your-key
python -m src.main process --input ./data/
```

### OpenAI
```bash
export OPENAI_API_KEY=your-key
python -m src.main process --provider openai --input ./data/
```

### Local Model (Ollama)
```bash
# Install Ollama first: https://ollama.com
ollama pull llama3.1:8b
python -m src.main process --provider ollama --model llama3.1:8b --input ./data/
```

---

## Commands

```bash
# Full analysis
python -m src.main process --input ./data/ --output ./outputs/

# Find warm paths to a company
python -m src.main warm-paths --target "Anthropic" --input ./data/

# Dry run (no LLM calls, shows what would be processed)
python -m src.main process --dry-run --input ./data/

# Specific output format only
python -m src.main process --format csv --input ./data/
```

---

## Output Files

| File | Description |
|------|-------------|
| `relationship_strength.csv` | Connections ranked by relationship strength |
| `reciprocity_ledger.csv` | Social capital balance (credit/debit) per person |
| `resurrection_candidates.md` | Dormant conversations worth reviving |
| `warm_paths_{company}.md` | Bridge candidates for target company |
| `network_summary.md` | Overall network analysis and insights |

---

## Configuration

Copy `config.yaml` to `config.local.yaml` for customization:

```yaml
# Key settings
llm:
  provider: anthropic  # or openai, ollama, vllm

relationship:
  decay_half_life_days: 180  # How fast relationships "cool"

reciprocity:
  scores:
    recommendation_written: 10  # Points for writing recommendations
```

See `config.yaml` for all options.

---

## How It Works

### Relationship Strength

Combines:
- **Recency** — When did you last interact?
- **Depth** — Were messages substantive or just "congrats!"?
- **Frequency** — How often do you message?
- **Reciprocity** — Two-way vs one-way relationship?

Uses exponential decay: strength halves every N days without interaction.

### Reciprocity Ledger

Tracks social capital:
- Writing a recommendation = +10 (you gave)
- Receiving a recommendation = -10 (you received)
- Net positive = you've invested more than received
- Net negative = you've received more than invested

Not a guilt tracker. It's awareness of relationship dynamics.

### Warm Paths

For a target company:
1. Find connections at similar/related companies
2. Rank by relationship strength
3. Output: "Reach out to X (strong connection) about Y (their context)"

---

## What This Won't Do

- ❌ Scrape LinkedIn
- ❌ Send automated messages
- ❌ Access real-time data
- ❌ Show your connections' connections
- ❌ Predict if someone will help you (we don't do "vouch scores")

These are intentional constraints, not limitations.

---

## Contributing

See `CONTRIBUTING.md` for guidelines.

Key files for developers:
- `CLAUDE.md` — Project context and architecture
- `CHECKLIST.md` — Development progress
- `DECISIONS.md` — Why things are built the way they are
- `LEARNINGS.md` — Gotchas and insights

---

## License

MIT License. See `LICENSE`.

---

## Acknowledgments

Built with:
- [Anthropic Claude](https://anthropic.com) / [OpenAI](https://openai.com) / [Ollama](https://ollama.com)
- [Pydantic](https://pydantic.dev)
- [Rich](https://rich.readthedocs.io)
