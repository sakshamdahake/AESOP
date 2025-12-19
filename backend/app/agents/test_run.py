import asyncio

from langchain_openai import ChatOpenAI

from app.agents.critic.agent import CriticAgent
from app.agents.critic.schemas import PaperGrade

# Mega-LLM configs (MATCH Scout & Critic node)
BASE_URL = "https://ai.megallm.io/v1"
AGENT_MODEL = "openai-gpt-oss-120b"


async def main():
    llm = ChatOpenAI(
        model=AGENT_MODEL,
        base_url=BASE_URL,
        temperature=0.2,
    )

    critic = CriticAgent(llm)

    research_question = (
        "Does metformin reduce all-cause mortality in patients with type 2 diabetes?"
    )

    # Minimal test abstracts (mocked but realistic)
    abstracts = [
        """
        This randomized controlled trial enrolled 1,200 patients with type 2 diabetes
        and followed them for 5 years. Patients treated with metformin showed a
        statistically significant reduction in all-cause mortality compared to
        standard therapy.
        """,
        """
        This case study reports outcomes in 12 patients with type 2 diabetes treated
        with metformin. Observational findings suggest possible benefit, but no
        control group was included.
        """,
    ]

    result = await critic.grade_batch(
        research_question=research_question,
        abstracts=abstracts,
    )

    print("\n=== CRITIC AGENT TEST ===")
    print("Decision:", result["decision"])
    print("\nGrades:")
    for i, grade in enumerate(result["grades"], start=1):
        print(f"\nPaper {i}:")
        print(grade.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
