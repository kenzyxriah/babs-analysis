from __future__ import annotations

import json
import re
import time
from typing import Any

import pandas as pd

from analytics.config.constants import INTENT_KEYWORDS, YES_WORDS, NO_WORDS


def normalize_text(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    return value.lower().strip()


def parse_yes_no(value: str | None) -> str:
    val = normalize_text(value)
    if not val:
        return "unknown"
    if any(word in val for word in YES_WORDS):
        return "yes"
    if any(word in val for word in NO_WORDS):
        return "no"
    return "unknown"


def parse_time_investment(value: str | None) -> str:
    val = normalize_text(value)
    if not val:
        return "time_unknown"
    digits = [int(x) for x in "".join(ch if ch.isdigit() else " " for ch in val).split() if x.isdigit()]
    hours = max(digits) if digits else None
    if hours is None:
        if "20+" in val or "20 +" in val:
            return "time_20_plus"
        return "time_unknown"
    if hours >= 20:
        return "time_20_plus"
    if hours >= 10:
        return "time_10_19"
    return "time_lt_10"


def extract_intent_tags(
    all_text: str,
    intent_raw: str | None,
    role_interest: str | None,
    time_investment: str | None,
    transition_answer: str | None,
    research_answer: str | None,
    readiness_answer: str | None,
) -> set[str]:
    text = normalize_text(all_text)
    tags: set[str] = set()

    for tag, keywords in INTENT_KEYWORDS.items():
        if any(k in text for k in keywords):
            tags.add(tag)

    intent_val = normalize_text(intent_raw)
    if "consult" in intent_val:
        tags.add("career_consultation")
    if "mentor" in intent_val:
        tags.add("mentorship")
    if "course" in intent_val:
        tags.add("course")
    if "interview" in intent_val:
        tags.add("interview_prep")

    transition_val = parse_yes_no(transition_answer)
    if transition_val == "yes":
        tags.add("transitioning")
    elif transition_val == "no":
        tags.add("not_transitioning")

    research_val = parse_yes_no(research_answer)
    if research_val == "yes":
        tags.add("researched_market")
    elif research_val == "no":
        tags.add("not_researched_market")

    readiness_val = parse_yes_no(readiness_answer)
    if readiness_val == "yes":
        tags.add("ready_committed")
    elif readiness_val == "no":
        tags.add("not_ready_committed")

    tags.add(parse_time_investment(time_investment))

    role_val = normalize_text(role_interest)
    if role_val:
        if "data science" in role_val or "data scientist" in role_val:
            tags.add("data_science")
        if "cyber" in role_val or "security" in role_val:
            tags.add("cybersecurity")
        if "cloud" in role_val or "aws" in role_val or "azure" in role_val:
            tags.add("cloud")

    return {tag for tag in tags if tag}


def parse_form_submissions(form_submissions: pd.DataFrame, forms: pd.DataFrame) -> pd.DataFrame:
    forms = forms[["id", "title"]].rename(columns={"id": "formId", "title": "formTitle"})

    rows = []
    for _, row in form_submissions.iterrows():
        data_raw = row.get("data")
        if not isinstance(data_raw, str):
            continue
        try:
            payload = json.loads(data_raw)
        except json.JSONDecodeError:
            continue

        contact = payload.get("contactInfo", {}) if isinstance(payload, dict) else {}
        sections = payload.get("sections", []) if isinstance(payload, dict) else []

        fields: dict[str, Any] = {}
        for section in sections:
            for key, value in (section.get("fields", {}) or {}).items():
                fields[key] = value

        intent_raw = None
        role_interest = None
        time_investment = None
        transition_answer = None
        research_answer = None
        readiness_answer = None
        current_skills = None
        target_role = None
        career_goal = None

        for k, v in fields.items():
            if not isinstance(k, str):
                continue
            key_lower = k.lower()
            if "precisely" in key_lower and "today" in key_lower:
                intent_raw = v
            if "specific it roles" in key_lower:
                role_interest = v
            if "how much time" in key_lower:
                time_investment = v
            if "transitioning" in key_lower:
                transition_answer = v
            if "researched job opportunities" in key_lower:
                research_answer = v
            if "financially and mentally prepared" in key_lower:
                readiness_answer = v
            if "current skills" in key_lower or "technical and soft skills" in key_lower:
                current_skills = v
            if "top 3 dream roles" in key_lower or "career path" in key_lower:
                target_role = v
            if "ultimate career goal" in key_lower:
                career_goal = v

        all_text = []
        for value in fields.values():
            if isinstance(value, str):
                all_text.append(value)
        for extra in [intent_raw, role_interest, time_investment]:
            if isinstance(extra, str):
                all_text.append(extra)

        intent_tags = extract_intent_tags(
            all_text=" ".join(all_text),
            intent_raw=intent_raw,
            role_interest=role_interest,
            time_investment=time_investment,
            transition_answer=transition_answer,
            research_answer=research_answer,
            readiness_answer=readiness_answer,
        )

        rows.append(
            {
                "submissionId": row.get("id"),
                "submittedAt": row.get("submittedAt"),
                "formId": row.get("formId"),
                "email": contact.get("email"),
                "firstName": contact.get("firstname"),
                "lastName": contact.get("lastname"),
                "phone": contact.get("yourwhatsappnumber"),
                "intentRaw": intent_raw,
                "roleInterest": role_interest,
                "timeInvestment": time_investment,
                "currentSkills": current_skills,
                "targetRole": target_role,
                "careerGoal": career_goal,
                "intentTags": ";".join(sorted(intent_tags)) if intent_tags else "",
            }
        )

    leads = pd.DataFrame(rows)
    leads = leads.merge(forms, on="formId", how="left")

    leads["intentCategory"] = "Other/Unknown"
    if not leads.empty:
        intent_raw_lower = leads["intentRaw"].fillna("").astype(str).str.lower()
        leads.loc[intent_raw_lower.str.contains("consult"), "intentCategory"] = "Career Consultation"
        leads.loc[intent_raw_lower.str.contains("mentor"), "intentCategory"] = "Mentorship"
        leads.loc[intent_raw_lower.str.contains("interview"), "intentCategory"] = "Interview Prep"
        leads.loc[intent_raw_lower.str.contains("course"), "intentCategory"] = "Course"

    return leads


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


async def extract_skill_gap_llm(
    leads: pd.DataFrame,
    groq_api_key: str | None,
    model: str,
    max_rows: int,
    batch_size: int,
    batch_sleep_seconds: int,
) -> pd.DataFrame:
    if not groq_api_key:
        return pd.DataFrame(
            {
                "submissionId": leads.get("submissionId"),
                "email": leads.get("email"),
                "status": "llm_skipped_no_key",
            }
        )

    try:
        from groq import AsyncGroq
    except Exception:
        return pd.DataFrame(
            {
                "submissionId": leads.get("submissionId"),
                "email": leads.get("email"),
                "status": "llm_skipped_missing_sdk",
            }
        )

    client = AsyncGroq(api_key=groq_api_key)

    rows = []
    batch_count = 0
    for _, row in leads.head(max_rows).iterrows():
        prompt = (
            "You are extracting structured data from a mentorship intake form. "
            "Return ONLY valid JSON with keys: career_goal_category, target_role_category, "
            "skills_list, skills_gap_list.\n\n"
            f"Career goal: {row.get('careerGoal')}\n"
            f"Target role: {row.get('targetRole')}\n"
            f"Current skills: {row.get('currentSkills')}\n"
        )
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = completion.choices[0].message.content if completion.choices else ""
            parsed = _extract_json(content)
        except Exception:
            parsed = {}

        rows.append(
            {
                "submissionId": row.get("submissionId"),
                "email": row.get("email"),
                "career_goal_category": parsed.get("career_goal_category"),
                "target_role_category": parsed.get("target_role_category"),
                "skills_list": parsed.get("skills_list"),
                "skills_gap_list": parsed.get("skills_gap_list"),
                "status": "ok" if parsed else "llm_failed",
            }
        )
        batch_count += 1
        if batch_count % batch_size == 0 and batch_count < min(max_rows, len(leads)):
            time.sleep(batch_sleep_seconds)

    return pd.DataFrame(rows)
