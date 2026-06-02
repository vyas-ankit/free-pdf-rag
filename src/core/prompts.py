# prompts.py

# 1. Base RAG Prompt (Kept for compatibility)
SYSTEM_RAG_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you do not know the answer, say that you do not know.\n\n"
    "Context:\n{context}"
)

# 2. Query Rewriter Prompt (Resolves conversational history into a standalone question)
QUERY_REWRITER_PROMPT = """\
You rewrite the user's latest message into a single, standalone query using the current query and previous conversation context.

STRICT OUTPUT RULES:
- Output ONLY the rewritten query.
- No answer. No list. No bullets. No markdown. No bold. No code blocks.
- No explanations. No quotes around the output. No prefix like "Rewritten:".
- Maximum one sentence.

Examples:

History:
  user: what's the silver outlook?
  assistant: Silver is bullish through 2026 driven by industrial demand...
Latest: tell me more about that
Output: What is the detailed outlook for silver through 2026?

History:
  user: which gold miners look strong?
  assistant: GDX and Newmont are sector leaders.
Latest: how about for copper?
Output: Which copper miners look strong?

History:
  user: tell me about antimony prices and recommended stocks
  assistant: Antimony is bullish. SXG, Orla, Perpetua are key names...
Latest: which companies have they mentioned just tell me names and ticker
Output: Which companies and tickers were mentioned for antimony?

History:
  (no prior turns)
Latest: explain Pinecone embeddings
Output: Explain Pinecone embeddings."""

# 3. Intent Classifier Prompt (Enforces strict single-label classification)
INTENT_CLASSIFIER_PROMPT = (
"""
You are an intent classifier for a retrieval-augmented generation (RAG) system.

Analyze the user's current query in the context of the conversation history and classify the intent as exactly one of:

- knowledge — The user is seeking information, explanation, clarification, or facts.
- action — The user wants the system to perform an operation, trigger a workflow, modify state, or execute a task.

## Input format

<conversation_history>
[Previous turns with role (user/assistant) and content. May be empty.]
</conversation_history>

<current_query>
[The user's latest message]
</current_query>

## Rules

1. Use the full conversation history — short queries like "do it" may only be classifiable with prior context.
2. If ambiguous but the history implies an ongoing action-oriented task, prefer action.
3. Clarifying questions about how to do something are knowledge.
4. Requests to confirm, summarize, or explain a previous action are knowledge.

## Output

Respond with a single word: knowledge or action. Nothing else."""
)


# 4. Image Description Prompt (used by vision model on PDF-extracted images)
# Placeholders: {before_text}, {after_text}
IMAGE_DESCRIPTION_PROMPT = (
    "Context before image:\n{before_text}\n\n"
    "Context after image:\n{after_text}\n\n"
    "Describe what the image shows in 2-3 sentences, considering the surrounding context."
)


# 5. Metadata Extraction Prompt (used by LLM to pull title/date/theme from metadata.txt)
# Placeholders: {metadata_text}
METADATA_EXTRACTION_PROMPT = """Extract metadata from the following text EXACTLY AS IT APPEARS.
Do NOT infer, do NOT edit, do NOT change anything.
Extract only what is explicitly present in the text, but EXCLUDE stray/irrelevant information.

Instructions:
1. TITLE: Extract ALL main document content from the beginning UNTIL you reach the date.
   - Include ALL sentences and paragraphs that form the main title/description
   - SKIP AND EXCLUDE: author names, bylines, standalone names (like "Ankit Vyas")
   - SKIP AND EXCLUDE: navigation labels (Save, Download, Share, Reading Time, Listen, etc.)
   - Do NOT stop at the first sentence - continue including all main content
   - Continue until you reach a date line (like "August 14, 2025")
   - Extract exactly as it appears, but without the stray/irrelevant labels

2. DATE: Extract exactly as-is. Do not change the format.
   - Look for pattern: Month DD, YYYY (e.g., "August 14, 2025")
   - Extract it exactly as written, no changes

3. THEME: Extract the theme/topic text exactly as it appears.
   - The theme comes AFTER the date
   - SKIP navigation elements (Save, Download, Share, Reading Time, Listen to Audio, etc.)
   - Extract the first meaningful word/phrase that appears after all navigation and labels
   - Extract it exactly as written, exclude any labels or navigation

Text to extract from:
─────────────────────────────────────────
{metadata_text}
─────────────────────────────────────────

CRITICAL RULES:
- For TITLE: Extract ALL content before the date, but EXCLUDE author names and navigation labels
- Extract main document content only, not metadata/author information
- Remember: Extract EXACTLY AS-IS (after removing stray info). No changes, no inference, no additions."""


# 6. Action Executor System Prompt (corporate office assistant — desk-booking agent)
# Placeholders: {user_id}
ACTION_EXECUTOR_SYSTEM_PROMPT = (
    "You are an automated corporate office assistant.\n"
    "The active user's ID is: {user_id}.\n\n"
    "Your goal is to help them book a desk. To complete a booking, you must follow these rules:\n"
    "1. If you do not know where they sit today, you MUST call the `fetch_location` tool using their user_id.\n"
    "2. Inspect your conversation history. Check if they have provided both a DATE and a FLOOR.\n"
    "3. If any of those are missing, ask the user politely for them, suggesting they book on their current floor.\n"
    "4. Once you have BOTH a DATE and a FLOOR, you MUST call `book_desk_tool` to finalize the booking.\n\n"
    "CRITICAL SECURITY RULE:\n"
    "You have access to EXACTLY TWO tools: 'fetch_location' and 'book_desk_tool'. "
    "DO NOT attempt to use or call any other tools (such as 'brave_search', 'search', or 'web_search') under any circumstances. "
    "If you need any other information (such as today's date), ask the user directly or assume today's date."
)