"""
LLM-based reranking with user preferences.

Uses an LLM (GPT-4o-mini by default) to rerank papers based on user-defined
research interests and preferences.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from paperfind.config import LLM_RERANK_MODEL, get_rerank_preferences
from paperfind.documents import extract_title_and_abstract
from paperfind.logging import get_logger

logger = get_logger(__name__)

DEFAULT_LLM_RERANK_MODEL = "gpt-4o-mini"

RERANK_PROMPT_TEMPLATE = """You are a research paper relevance scorer. Score each paper from 0 to 10 based on how relevant it is to the user's research interests.

USER PREFERENCES:
{preferences}
{keyword_section}
PAPERS TO SCORE:
{papers_list}

For each paper, return a JSON object with "id" (the paper number) and "score" (0-10, where 10 is most relevant).
Output ONLY a valid JSON array, no explanation or other text.

Example output format:
[{{"id": 1, "score": 8}}, {{"id": 2, "score": 3}}, {{"id": 3, "score": 9}}]"""


def _format_paper_for_prompt(idx: int, doc: Document) -> str:
    """Format a single paper for the LLM prompt."""
    title, abstract = extract_title_and_abstract(doc)
    source = doc.metadata.get("source", "unknown")

    return f"""Paper {idx}:
Title: {title}
Source: {source}
Abstract: {abstract or 'No abstract available'}
"""


def _call_openai(prompt: str, model: str) -> str:
    """Call OpenAI API to get reranking scores."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "LLM reranking requires openai. Install with: pip install openai"
        ) from exc

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2000,
    )
    return response.choices[0].message.content or ""


def _parse_scores(response: str, num_papers: int) -> Dict[int, float]:
    """Parse LLM response into paper scores."""
    scores: Dict[int, float] = {}

    # Clean up response - sometimes LLM adds markdown code blocks
    response = response.strip()
    if response.startswith("```"):
        # Remove markdown code block
        lines = response.split("\n")
        response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        response = response.strip()

    try:
        data = json.loads(response)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "id" in item and "score" in item:
                    paper_id = int(item["id"])
                    score = float(item["score"])
                    # Clamp score to 0-10 range
                    score = max(0, min(10, score))
                    scores[paper_id] = score
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse LLM response: {e}")
        logger.debug(f"Response was: {response[:500]}")

    # Fill in missing scores with default
    for i in range(1, num_papers + 1):
        if i not in scores:
            scores[i] = 5.0  # Neutral score for unparsed papers

    return scores


def llm_rerank(
    papers: List[Tuple[str, Document, str]],
    preferences: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    model: Optional[str] = None,
    batch_size: int = 20,
) -> List[Tuple[str, float, Document, str]]:
    """
    Rerank papers using an LLM with user preferences.

    Args:
        papers: List of (doi, document, source_info) tuples
        preferences: User preferences text. If None, loads from config.
        keywords: Optional keyword phrases to include in prompt
        model: LLM model to use. If None, uses LLM_RERANK_MODEL config.
        batch_size: Number of papers to process per API call

    Returns:
        List of (doi, score, document, source_info) tuples sorted by score descending.
        Score is 0-10 where 10 is most relevant.

    Raises:
        ValueError: If no preferences are configured
    """
    if not preferences:
        preferences = get_rerank_preferences()

    if not preferences:
        raise ValueError(
            "No LLM rerank preferences configured. "
            "Set LLM_RERANK_PREFERENCES env var or create "
            "~/.paperfind/rerank_preferences.txt"
        )

    model = model or os.getenv("LLM_RERANK_MODEL", LLM_RERANK_MODEL)
    logger.info(f"LLM reranking {len(papers)} papers with {model}")

    # Build keyword section for prompt
    keyword_section = ""
    if keywords:
        keyword_section = f"\nCURRENT SEARCH KEYWORDS: {', '.join(keywords)}\n"

    # Process in batches
    all_scores: Dict[str, Tuple[float, Document, str]] = {}

    for batch_start in range(0, len(papers), batch_size):
        batch = papers[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(papers) + batch_size - 1) // batch_size

        logger.debug(f"Processing batch {batch_num}/{total_batches}")

        # Format papers for prompt
        papers_list = "\n".join(
            _format_paper_for_prompt(i + 1, doc) for i, (_, doc, _) in enumerate(batch)
        )

        prompt = RERANK_PROMPT_TEMPLATE.format(
            preferences=preferences,
            keyword_section=keyword_section,
            papers_list=papers_list,
        )

        # Call LLM
        response = _call_openai(prompt, model)
        scores = _parse_scores(response, len(batch))

        # Map scores back to DOIs
        for i, (doi, doc, source_info) in enumerate(batch):
            paper_id = i + 1
            score = scores.get(paper_id, 5.0)
            # Keep best score if we've seen this DOI before
            if doi not in all_scores or score > all_scores[doi][0]:
                all_scores[doi] = (score, doc, source_info)

    # Sort by score descending (higher is more relevant)
    sorted_results = sorted(all_scores.items(), key=lambda x: x[1][0], reverse=True)

    return [(doi, score, doc, source) for doi, (score, doc, source) in sorted_results]


def llm_rerank_candidates(
    candidates: List[Tuple[str, Tuple[float, Document, str, str]]],
    preferences: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    k: int = 10,
) -> Tuple[List[Tuple[str, Tuple[float, Document, str]]], bool]:
    """
    Rerank recommendation candidates using LLM.

    This is a convenience wrapper that matches the interface expected by
    recommend.py's reranking flow.

    Args:
        candidates: List of (doi, (score, doc, source_info, query_text)) tuples
        preferences: User preferences text
        keywords: Optional keyword phrases
        k: Number of results to return

    Returns:
        Tuple of (reranked_results, rerank_used) where:
        - reranked_results: List of (doi, (score, doc, source_info)) tuples
        - rerank_used: True if reranking was performed
    """
    if not candidates:
        return [], False

    # Extract papers for reranking
    papers = [(doi, doc, source) for doi, (_, doc, source, _) in candidates]

    try:
        reranked = llm_rerank(
            papers=papers,
            preferences=preferences,
            keywords=keywords,
        )

        # Convert back to expected format, take top k
        results = [(doi, (score, doc, source)) for doi, score, doc, source in reranked[:k]]
        return results, True

    except Exception as e:
        logger.error(f"LLM reranking failed: {e}")
        # Fall back to original order
        results = [(doi, (score, doc, source)) for doi, (score, doc, source, _) in candidates[:k]]
        return results, False


GENERATE_PREFERENCES_PROMPT = """Based on the following context about a researcher's interests, generate a preferences file for paper recommendations.

CONTEXT:
{context}

Generate a preferences file with:
1. An "I'm interested in:" section with 4-6 specific bullet points based on the context
2. An "I'm NOT interested in:" section with 3-4 bullet points (infer what's likely NOT relevant)
3. Any relevant notes about the researcher's focus

Be specific and detailed based on the context provided. Output ONLY the preferences text, no explanations or markdown formatting."""


def generate_preferences(
    keywords: Optional[List[str]] = None,
    collection: Optional[str] = None,
    zotero_titles: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> str:
    """
    Generate an initial preferences file based on context.

    Args:
        keywords: Optional keyword phrases the user is interested in
        collection: Optional Zotero collection name
        zotero_titles: Optional list of paper titles from user's library
        model: LLM model to use. If None, uses LLM_RERANK_MODEL config.

    Returns:
        Generated preferences text

    Raises:
        ValueError: If no context is provided
    """
    context_parts = []

    if keywords:
        context_parts.append(f"Search keywords: {', '.join(keywords)}")

    if collection:
        context_parts.append(f"Zotero collection name: '{collection}'")

    if zotero_titles:
        titles_sample = zotero_titles[:20]  # Limit to 20 titles
        titles_text = "\n".join(f"- {t}" for t in titles_sample)
        context_parts.append(f"Sample papers from their library:\n{titles_text}")

    if not context_parts:
        raise ValueError(
            "No context provided. Use --keywords, --collection, or sync your Zotero library first."
        )

    context = "\n\n".join(context_parts)
    model = model or os.getenv("LLM_RERANK_MODEL", LLM_RERANK_MODEL)

    logger.info(f"Generating preferences with {model}...")

    prompt = GENERATE_PREFERENCES_PROMPT.format(context=context)
    response = _call_openai(prompt, model)

    # Clean up response
    preferences = response.strip()

    # Remove markdown code blocks if present
    if preferences.startswith("```"):
        lines = preferences.split("\n")
        preferences = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        preferences = preferences.strip()

    return preferences


UPDATE_PREFERENCES_PROMPT = """You are helping a researcher refine their paper recommendation preferences based on feedback they've provided on past recommendations.

CURRENT PREFERENCES:
{current_preferences}

USER FEEDBACK ON PAST RECOMMENDATIONS:

{feedback_section}

Based on this feedback, generate an UPDATED preferences file that:
1. Keeps what still applies from the current preferences
2. Incorporates patterns from the "liked" and "example" papers (what the user IS looking for)
3. Adds exclusions based on "disliked" papers and their reasons (what the user is NOT looking for)
4. Be specific about what distinguishes good matches from superficial similarities

Output ONLY the updated preferences text in the same format as the current preferences (with "I'm interested in:" and "I'm NOT interested in:" sections). Do not include any explanations or markdown formatting."""


def update_preferences_from_feedback(
    current_preferences: str,
    feedback: List[Dict],
    model: Optional[str] = None,
) -> str:
    """
    Generate updated preferences based on accumulated user feedback.

    Args:
        current_preferences: Current preferences text
        feedback: List of feedback dicts with doi, feedback_type, reason,
                  paper_title, paper_abstract
        model: LLM model to use. If None, uses LLM_RERANK_MODEL config.

    Returns:
        Updated preferences text

    Raises:
        ValueError: If no feedback is provided
    """
    if not feedback:
        raise ValueError("No feedback to process")

    # Group feedback by type
    likes = [f for f in feedback if f.get("feedback_type") == "like"]
    dislikes = [f for f in feedback if f.get("feedback_type") == "dislike"]
    examples = [f for f in feedback if f.get("feedback_type") == "example"]

    # Format feedback section
    feedback_parts = []

    if examples:
        feedback_parts.append("STRONG POSITIVE EXAMPLES (papers the user specifically wants more like):")
        for f in examples:
            title = f.get("paper_title") or f.get("doi")
            feedback_parts.append(f'- "{title}"')
            if f.get("reason"):
                feedback_parts.append(f"  User's reason: {f['reason']}")

    if likes:
        feedback_parts.append("\nLIKED PAPERS (relevant to user's interests):")
        for f in likes:
            title = f.get("paper_title") or f.get("doi")
            feedback_parts.append(f'- "{title}"')
            if f.get("reason"):
                feedback_parts.append(f"  User's reason: {f['reason']}")

    if dislikes:
        feedback_parts.append("\nDISLIKED PAPERS (NOT relevant, even if superficially similar):")
        for f in dislikes:
            title = f.get("paper_title") or f.get("doi")
            feedback_parts.append(f'- "{title}"')
            if f.get("reason"):
                feedback_parts.append(f"  User's reason: {f['reason']}")

    feedback_section = "\n".join(feedback_parts)

    model = model or os.getenv("LLM_RERANK_MODEL", LLM_RERANK_MODEL)
    logger.info(f"Updating preferences from {len(feedback)} feedback entries using {model}...")

    prompt = UPDATE_PREFERENCES_PROMPT.format(
        current_preferences=current_preferences,
        feedback_section=feedback_section,
    )

    response = _call_openai(prompt, model)

    # Clean up response
    preferences = response.strip()

    # Remove markdown code blocks if present
    if preferences.startswith("```"):
        lines = preferences.split("\n")
        preferences = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        preferences = preferences.strip()

    return preferences


# =============================================================================
# Simplified single-feedback preference update
# =============================================================================

SIMPLE_FEEDBACK_PROMPT = """You are helping a researcher refine their paper recommendation preferences.

CURRENT PREFERENCES:
{current_preferences}

USER FEEDBACK:
The user wants to {action}: {reason}

Update the preferences to incorporate this feedback:
- If "like": Add or strengthen relevant interests in the "I'm interested in:" section
- If "dislike": Add or strengthen relevant exclusions in the "I'm NOT interested in:" section
- Keep existing preferences that still apply
- Be specific about what the user is describing

Output ONLY the updated preferences text in the same format (with "I'm interested in:" and "I'm NOT interested in:" sections). Do not include any explanations or markdown formatting."""


def update_preferences_with_feedback(
    current_preferences: str,
    feedback_type: str,
    reason: str,
    model: Optional[str] = None,
) -> str:
    """
    Update preferences based on a single feedback item.

    This is a simplified version that directly updates preferences without
    storing feedback in a database.

    Args:
        current_preferences: Current preferences text
        feedback_type: "like" or "dislike"
        reason: User's description of what they like/dislike
        model: LLM model to use. If None, uses LLM_RERANK_MODEL config.

    Returns:
        Updated preferences text
    """
    action = "see MORE papers like this" if feedback_type == "like" else "AVOID papers like this"

    model = model or os.getenv("LLM_RERANK_MODEL", LLM_RERANK_MODEL)
    logger.info(f"Updating preferences with {feedback_type} feedback using {model}...")

    prompt = SIMPLE_FEEDBACK_PROMPT.format(
        current_preferences=current_preferences,
        action=action,
        reason=reason,
    )

    response = _call_openai(prompt, model)

    # Clean up response
    preferences = response.strip()

    # Remove markdown code blocks if present
    if preferences.startswith("```"):
        lines = preferences.split("\n")
        preferences = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        preferences = preferences.strip()

    return preferences
