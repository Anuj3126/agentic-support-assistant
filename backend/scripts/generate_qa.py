"""One-time data generation script — NOT part of the runtime package.

Generates synthetic FAQ datasets per department using Gemini and writes them into
shopunow/data/qa/<department>.json, where the app loads them from at index-build time.

Run from backend/: `python scripts/generate_qa.py`
Safe to re-run — skips departments that already have a file, pass --force to regenerate all.
Once shopunow/data/qa/*.json exists and the Chroma index is built, this script (and this
whole scripts/ folder) is no longer needed at runtime.
"""

import argparse
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from shopunow.config import DATA_DIR, settings
from shopunow.data.departments import DEPARTMENTS

OUT_DIR = DATA_DIR / "qa"

PROMPT_TEMPLATE = """Act as an expert running the {label} department for Company ShopUNow,
a retail company selling a wide variety of products around clothing, DIY (do it yourself),
books, and toys.

{description}

This department serves {audience} users ({audience_note}).

Create a list of {n} questions and answers which could be the most frequently asked questions
for this department. Return each as an object with a 'doc' field containing the question and
answer together as a single string formatted as "Q: ...\\nA: ..." and a 'category' field set to
"{dept_id}"."""


class QAPair(BaseModel):
    doc: str
    category: str


class QAList(BaseModel):
    items: list[QAPair]


def audience_note(audience: str) -> str:
    return (
        "ShopUNow's own employees" if audience == "internal" else "external ShopUNow customers"
    )


def generate_for_department(llm: ChatGoogleGenerativeAI, dept_id: str, n: int = 15) -> list[dict]:
    dept = DEPARTMENTS[dept_id]
    prompt = PROMPT_TEMPLATE.format(
        label=dept.label,
        description=dept.description,
        audience=dept.audience,
        audience_note=audience_note(dept.audience),
        n=n,
        dept_id=dept.id,
    )
    structured_llm = llm.with_structured_output(QAList)
    result: QAList = structured_llm.invoke(prompt)
    return [{"id": f"{dept_id}-{i}", **item.model_dump()} for i, item in enumerate(result.items)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="regenerate even if file exists")
    parser.add_argument("--n", type=int, default=15, help="QA pairs per department")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    llm = ChatGoogleGenerativeAI(model=settings.chat_model, google_api_key=settings.google_api_key)

    for dept_id in DEPARTMENTS:
        out_path = OUT_DIR / f"{dept_id}.json"
        if out_path.exists() and not args.force:
            print(f"skip {dept_id} (already exists)")
            continue
        print(f"generating {args.n} QA pairs for {dept_id}...")
        qa_pairs = generate_for_department(llm, dept_id, n=args.n)
        out_path.write_text(json.dumps(qa_pairs, indent=2))
        print(f"wrote {len(qa_pairs)} pairs to {out_path}")


if __name__ == "__main__":
    main()
