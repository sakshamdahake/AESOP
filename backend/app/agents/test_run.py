from langchain_openai import ChatOpenAI

from app.agents.critic.agent import CriticAgent

# ============================
# Mega-LLM configuration
# ============================

# BASE_URL = "https://ai.megallm.io/v1"
# AGENT_MODEL = "openai-gpt-oss-120b"

# ============================
# AWS Bedrock configuration
# ============================

base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1", 
api_key="$AWS_BEARER_TOKEN_BEDROCK"


# ============================
# Mock retrieval batches
# ============================

def retrieve_batch(iteration: int) -> list[str]:
    """
    Simulates improving retrieval quality across CRAG iterations.
    """

    if iteration == 0:
        return [
            """
            This case series reports outcomes in 10 patients with type 2 diabetes
            treated with metformin. No control group was included.
            """
        ]

    if iteration == 1:
        return [
            """
            This observational cohort study followed 180 patients with type 2 diabetes
            treated with metformin for 2 years and compared mortality outcomes.
            """
        ]

    return [
        """
        This randomized controlled trial enrolled 1,200 patients with type 2 diabetes
        and followed them for 5 years. Metformin treatment resulted in a statistically
        significant reduction in all-cause mortality.
        """
    ]


# ============================
# CRAG loop test (SYNC)
# ============================

def main():
    llm = ChatOpenAI(
        base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1", 
        api_key="$AWS_BEARER_TOKEN_BEDROCK"
    )

    critic = CriticAgent(llm)

    research_question = (
        "Does metformin reduce all-cause mortality in patients with type 2 diabetes?"
    )

    max_iterations = 3
    iteration = 0

    print("\n=== CRAG LOOP TEST (SYNC CRITIC) ===")

    while iteration < max_iterations:
        print(f"\n--- Iteration {iteration + 1} ---")

        abstracts = retrieve_batch(iteration)

        result = critic.grade_batch(
            research_question=research_question,
            abstracts=abstracts,
            iteration=iteration,
        )

        print("Decision:", result["decision"])

        for i, grade in enumerate(result["grades"], start=1):
            print(f"\nPaper {i}:")
            print(grade.model_dump())

        if result["decision"] == "sufficient":
            print("\n✅ Evidence sufficient — stopping CRAG loop.")
            break

        print("\n🔄 Evidence insufficient — retrieving more...")
        iteration += 1

    else:
        print("\n⚠️ Max iterations reached without sufficient evidence.")


if __name__ == "__main__":
    main()
