import os
import re
import math
import difflib
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


# ============================================================
# SECTION 1: SUBJECT & CITATION CONFIGURATION
# ============================================================

TASK_BASE_SCALARS = {
    "rewrite_light":    {"lex": 20, "order": 10},
    "rewrite_medium":   {"lex": 30, "order": 20},
    "rewrite_heavy":    {"lex": 40, "order": 35},
    "rewrite_complete": {"lex": 55, "order": 55},
    "paraphrase_deep":  {"lex": 60, "order": 20}
}

SUBJECT_MODIFIERS = {
    "engineering":    {"lex_offset": 0,   "order_cap": 25},
    "legal":          {"lex_offset": -5,  "order_cap": 15},
    "humanities":     {"lex_offset": +5,  "order_cap": 60},
    "social_science": {"lex_offset": 0,   "order_cap": 45},
    "business":       {"lex_offset": 0,   "order_cap": 40}
}

SUBJECT_TO_CITATIONS = {
    "humanities":     ["mla", "chicago_author_date", "apa"],
    "social_science": ["apa", "harvard"],
    "engineering":    ["ieee"],
    "business":       ["harvard", "apa"],
    "legal":          ["oscola"]
}


# ============================================================
# SECTION 2: TOKEN BUDGET
# ============================================================

MAX_DEVELOPER_TOKENS = 565
CONTEXT_REFRESH_TOKENS = 150
SYSTEM_PROMPT_TOKENS = 35
SAFETY_ALLOWANCE = 60
TARGET_SLICE_MAX = MAX_DEVELOPER_TOKENS - (CONTEXT_REFRESH_TOKENS + SYSTEM_PROMPT_TOKENS + SAFETY_ALLOWANCE)

OVERLAP_SENTENCES = 2


# ============================================================
# SECTION 3: HELPER FUNCTIONS
# ============================================================

def split_sentences(text: str) -> list:
    text = text.replace("\n", " [NEWLINE] ")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s:
            s = s.replace("[NEWLINE]", "\n")
            cleaned.append(s)
    return cleaned

def get_last_sentences(text: str, n: int) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= n:
        return text
    return " ".join(sentences[-n:])

def get_first_sentences(text: str, n: int) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= n:
        return text
    return " ".join(sentences[:n])

def trim_overlap(prev_output: str, current_output: str, overlap_sentences: int = 2) -> str:
    if not prev_output or not current_output:
        return current_output

    prev_end = get_last_sentences(prev_output, overlap_sentences)
    current_start = get_first_sentences(current_output, overlap_sentences)

    prev_words = prev_end.split()
    curr_words = current_start.split()

    matcher = difflib.SequenceMatcher(None, prev_words, curr_words)
    match = matcher.find_longest_match(0, len(prev_words), 0, len(curr_words))
    overlap_len = match.size

    if overlap_len == 0:
        return current_output

    all_words = current_output.split()
    trimmed_words = all_words[overlap_len:]
    return " ".join(trimmed_words)

def clean_dipper_output(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(r"\[Current Content to Process\]:", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[Summary of Batch \d+:.*?\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", cleaned)
    return cleaned.strip()


# ============================================================
# SECTION 4: VALIDATION
# ============================================================

def validate_inputs(selected_task: str, selected_subject: str, selected_citation: str):
    errors = []
    if selected_task not in TASK_BASE_SCALARS:
        errors.append(f"Invalid task type: '{selected_task}'")
    if selected_subject not in SUBJECT_MODIFIERS:
        errors.append(f"Invalid subject discipline: '{selected_subject}'")
    if selected_subject in SUBJECT_TO_CITATIONS:
        if selected_citation not in SUBJECT_TO_CITATIONS[selected_subject]:
            errors.append(f"Citation style '{selected_citation}' is invalid for discipline '{selected_subject}'")
    else:
        errors.append(f"Subject '{selected_subject}' has no valid citation styles")
    return errors


# ============================================================
# SECTION 5: MODEL LOADING
# ============================================================

TOKENIZER_ID = "google/t5-v1_1-xxl"
MODEL_DIR = "/runpod-volume/dipper-model"

MODEL_DTYPE = torch.bfloat16


print("[ModelStore] Loading tokenizer and model...")

tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_ID)

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=MODEL_DTYPE,
    device_map="auto",
    local_files_only=True,
)
model.eval()
print("[ModelStore] Model loaded successfully.")


# ============================================================
# SECTION 6: CORE DIPPER PIPELINE
# ============================================================

progress_store = {}


def run_dipper_pipeline(
    user_manuscript: str,
    lexical_slider: int = 60,
    order_slider: int = 60,
    selected_task: str = "paraphrase_deep",
    selected_subject: str = "social_science",
    selected_citation: str = "apa",
    job_id: str = None
):
    errors = validate_inputs(selected_task, selected_subject, selected_citation)
    if errors:
        return {"error": errors}

    if job_id:
        progress_store[job_id] = {"status": "RECEIVED", "progress": 0, "message": "Job received. Starting..."}

    base = TASK_BASE_SCALARS[selected_task]
    modifier = SUBJECT_MODIFIERS[selected_subject]
    final_lex = min(60, max(0, base["lex"] + modifier["lex_offset"]))
    final_order = min(modifier["order_cap"], base["order"])

    lex_code = int(100 - final_lex)
    order_code = int(100 - final_order)

    if job_id:
        progress_store[job_id] = {"status": "INITIALIZING", "progress": 0, "message": "Initializing model and splitting document..."}

    raw_sentences = split_sentences(user_manuscript)
    batches = []
    current_batch_tokens = []
    current_batch_text = []

    for sentence in raw_sentences:
        if not sentence.strip():
            continue
        sentence_tokens = tokenizer.tokenize(sentence)
        if len(current_batch_tokens) + len(sentence_tokens) <= TARGET_SLICE_MAX:
            current_batch_tokens.extend(sentence_tokens)
            current_batch_text.append(sentence)
        else:
            if current_batch_text:
                batches.append(" ".join(current_batch_text))
            current_batch_tokens = list(sentence_tokens)
            current_batch_text = [sentence]

    if current_batch_text:
        batches.append(" ".join(current_batch_text))

    total_batches = len(batches)

    humanized_chunks = []
    context_buffer = ""

    for index, current_target_text in enumerate(batches):
        current_step = index + 1

        if job_id:
            progress_store[job_id] = {
                "status": "PROCESSING",
                "progress": round((current_step / total_batches) * 100, 2),
                "current_batch": current_step,
                "total_batches": total_batches,
                "message": f"Processing batch {current_step} of {total_batches}"
            }

        prompt = f"lexical = {lex_code}, order = {order_code}"
        if context_buffer:
            prompt += f" {context_buffer}"
        prompt += f" <sent> {current_target_text} </sent>"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_DEVELOPER_TOKENS,
        ).to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                top_p=0.75,
                top_k=None,
                max_length=512,
            )

        paraphrased_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if index == 0:
            humanized_chunks.append(paraphrased_output)
        else:
            trimmed = trim_overlap(humanized_chunks[-1], paraphrased_output, OVERLAP_SENTENCES)
            humanized_chunks.append(trimmed)

        output_tokens = tokenizer.tokenize(paraphrased_output)
        trailing_tokens = output_tokens[-CONTEXT_REFRESH_TOKENS:] if len(output_tokens) > CONTEXT_REFRESH_TOKENS else output_tokens
        context_buffer = tokenizer.convert_tokens_to_string(trailing_tokens)

    raw_text = " ".join(humanized_chunks).replace(" .", ".").replace(" \n ", "\n")
    complete_text = clean_dipper_output(raw_text)

    original_words = user_manuscript.split()
    humanized_words = complete_text.split()
    matcher = difflib.SequenceMatcher(None, original_words, humanized_words)

    total_words_changed = 0
    diff_data = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('replace', 'delete', 'insert'):
            total_words_changed += max(i2 - i1, j2 - j1)
        diff_data.append({
            "tag": tag,
            "original_start": i1,
            "original_end": i2,
            "humanized_start": j1,
            "humanized_end": j2,
            "original_segment": " ".join(original_words[i1:i2]) if i1 < i2 else "",
            "humanized_segment": " ".join(humanized_words[j1:j2]) if j1 < j2 else ""
        })

    percentage_altered = (total_words_changed / max(len(original_words), 1)) * 100

    if job_id:
        progress_store[job_id] = {"status": "COMPLETED", "progress": 100, "message": "Complete!"}

    return {
        "status": "success",
        "processed_batches": total_batches,
        "task_type": selected_task,
        "subject_discipline": selected_subject,
        "citation_style": selected_citation,
        "lexical_diversity": final_lex,
        "order_diversity": final_order,
        "metrics": {
            "original_word_count": len(original_words),
            "final_word_count": len(humanized_words),
            "estimated_words_altered": total_words_changed,
            "percentage_altered": round(percentage_altered, 2)
        },
        "diff_data": diff_data,
        "output_text": complete_text
    }


# ============================================================
# SECTION 7: RUNPOD SERVERLESS HANDLER
# ============================================================

def dipper_handler(job):
    try:
        input_data = job.get("input", {})
        manuscript = input_data.get("manuscript")
        if not manuscript:
            return {"error": "Missing 'manuscript' field"}

        selected_task = input_data.get("task", "paraphrase_deep")
        selected_subject = input_data.get("subject", "social_science")
        selected_citation = input_data.get("citation", "apa")
        lexical = input_data.get("lexical", None)
        order = input_data.get("order", None)

        job_id = job.get("id", None)

        if lexical is None or order is None:
            result = run_dipper_pipeline(
                manuscript,
                lexical_slider=60 if lexical is None else lexical,
                order_slider=60 if order is None else order,
                selected_task=selected_task,
                selected_subject=selected_subject,
                selected_citation=selected_citation,
                job_id=job_id
            )
        else:
            result = run_dipper_pipeline(
                manuscript,
                lexical_slider=lexical,
                order_slider=order,
                selected_task=selected_task,
                selected_subject=selected_subject,
                selected_citation=selected_citation,
                job_id=job_id
            )

        return result
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# SECTION 8: START
# ============================================================

    import runpod
    runpod.serverless.start({"handler": dipper_handler})
