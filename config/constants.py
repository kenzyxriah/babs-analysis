from __future__ import annotations

GATEWAY_KEYWORDS = ["interview prep", "interview", "prep"]
MENTORSHIP_KEYWORDS = ["mentorship", "mentor", "coaching"]
GATEWAY_SESSION_KEYWORDS = [
    "info session",
    "information session",
    "intro",
    "introduction",
    "orientation",
    "webinar",
    "open day",
    "open-day",
    "masterclass",
    "taster",
    "trial",
    "interview",
    "interview prep",
    "prep",
]
GATEWAY_PRODUCT_KEYWORDS = list(
    {
        *GATEWAY_KEYWORDS,
        "intro",
        "starter",
        "foundation",
        "taster",
        "trial",
        "basic",
        "entry",
        "one-time",
        "a la carte",
        "ala carte",
    }
)

INTENT_KEYWORDS = {
    "career_consultation": ["consult", "consultation", "career guidance", "career advice"],
    "mentorship": ["mentorship", "mentor", "coaching"],
    "course": ["course", "class", "training", "bootcamp", "program"],
    "interview_prep": ["interview", "mock interview", "interview prep"],
    "resume_cv": ["resume", "cv", "portfolio"],
    "job_search": ["job", "role", "apply", "application", "hiring"],
    "certification": ["certification", "certificate", "certified"],
    "cybersecurity": ["cyber", "security", "soc", "siem"],
    "data_science": ["data science", "data scientist", "machine learning", "ml"],
    "data_analytics": ["data analyst", "data analytics", "analytics", "bi"],
    "cloud": ["aws", "azure", "gcp", "cloud"],
    "devops": ["devops", "kubernetes", "docker", "ci/cd"],
    "software_engineering": ["software", "developer", "programming", "full stack", "backend", "frontend"],
    "product_management": ["product manager", "product management", "pm"],
    "project_management": ["project management", "scrum", "agile", "pmp"],
    "ui_ux": ["ux", "ui", "design"],
}

YES_WORDS = ["yes", "yep", "yeah", "y"]
NO_WORDS = ["no", "nope", "nah", "n"]
