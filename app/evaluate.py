from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from app.agent import run_agent
from app.retriever import retrieve
import asyncio
import json

TEST_SET = [
    {
        "question": "What is the main topic of the document?",
        "ground_truth": "The document covers the main subject matter uploaded by the user."
    },
    {
        "question": "What are the key points mentioned?",
        "ground_truth": "The key points are the main arguments or findings in the document."
    },
]

async def run_evaluation():
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_SET:
        result = await run_agent(item["question"])
        docs = retrieve(item["question"])

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([d["text"] for d in docs])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )

    print(json.dumps(scores.to_pandas().mean().to_dict(), indent=2))
    return scores

if __name__ == "__main__":
    asyncio.run(run_evaluation())