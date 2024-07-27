# ChatBot Evaluation and Improvement

## ⚠️Instructions

1. **Paste your HuggingFace API token** in the [.env](./.env) file.
2. **Install the requirements** using:
    ```sh
    pip install -r requirements.txt
    ```
3. **Download the default English language model** for SpaCy by running:
    ```sh
    python -m spacy download en
    ```

## File Structure

- **metrics**/
  - [answer_correctness.py](metrics/answer_correctness.py): Code for calculating the `answer_correctness` metric.
  - [answer_similarity.py](metrics/answer_similarity.py): Code for calculating the `answer_similarity` metric.
  - [faithfullness.py](metrics/faithfullness.py): Code for calculating the `faithfulness` metric.
  - [relevance.py](metrics/relevance.py): Code for calculating the `relevance` metric.
  
- [scores/](scores) Directory containing the calculated score .csv files.
- [.env](.env): Environment file for storing the HuggingFace API token.
- [aws_faq_chatbot.py](aws_faq_chatbot.py): The chatbot itself.
- [evaluator.ipynb](evaluator.ipynb): Jupyter notebook containing all the steps and processes of evaluating the chatbot.
- [evaluator_report.pdf](evaluator_report.pdf): PDF report of the evaluation.
- [README.md](README.md): This README file.
- [requirements.txt](requirements.txt): Python libraries to install.

## Evaluation Mechanism

For evaluating the responses of our ChatBot, I have established an evaluation mechanism centered around four key metrics:
- **Answer Similarity**
- **Faithfulness**
- **Relevance**
- **Answer Correctness**

### How This Works

We have created an LLM Evaluator for each metric, incorporating specific evaluation criteria and a scoring rubric ranging from 1 to 5 (where 1 denotes the lowest score and 5 denotes the highest score).

This criteria and scoring rubric are provided to an Evaluator LLM along with the generated response and the retrieved context from which the response is generated. After processing the input, the Evaluator LLM outputs:
- A numeric score between 1 and 5
- Feedback explaining why this particular score was given

Each metric and its scoring criteria are defined in detail later in the report.

## ChatBot Evaluation

### Answer Similarity
**Question:** Is the response semantically similar based on the context?

**Scoring Criteria:**
- **Score 1:** The response has little to no semantic similarity to the reference answer.
- **Score 2:** The response displays partial semantic similarity to the reference answer on some aspects.
- **Score 3:** The response has moderate semantic similarity to the reference answer.
- **Score 4:** The response aligns with the reference answer in most aspects and has substantial semantic similarity.
- **Score 5:** The response closely aligns with the reference answer in all significant aspects.

### Faithfulness
**Question:** Does the response accurately reflect the given context?

**Scoring Criteria:**
- **Score 1:** None of the claims in the response can be inferred from the provided context.
- **Score 2:** Some of the claims in the response can be inferred from the provided context, but the majority of the response is missing from, inconsistent with, or contradictory to the provided context.
- **Score 3:** Half or more of the claims in the response can be inferred from the provided context.
- **Score 4:** Most of the claims in the response can be inferred from the provided context, with very little information that is not directly supported by the provided context.
- **Score 5:** All of the claims in the response are directly supported by the provided context, demonstrating high faithfulness to the provided context.

### Relevance
**Question:** Does the response address the given question and context?

**Scoring Criteria:**
- **Score 1:** The response doesn't mention anything about the question or is completely irrelevant to the provided context.
- **Score 2:** The response provides some relevance to the question and is somehow related to the provided context.
- **Score 3:** The response mostly answers the question and is largely consistent with the provided context.
- **Score 4:** The response answers the question and is consistent with the provided context.
- **Score 5:** The response answers the question comprehensively using the provided context.

### Answer Correctness
**Question:** Is the response pertinent to the question and context?

**Scoring Criteria:**
- **Score 1:** The response is completely incorrect. It is completely different from or contradicts the provided target.
- **Score 2:** The response demonstrates some degree of semantic similarity and includes partially correct information. However, the response still has significant discrepancies with the provided target or inaccuracies.
- **Score 3:** The response addresses a couple of aspects of the input accurately, aligning with the provided target. However, there are still omissions or minor inaccuracies.
- **Score 4:** The response is mostly correct. It provides mostly accurate information, but there may be one or more minor omissions or inaccuracies.
- **Score 5:** The response is correct. It demonstrates a high degree of accuracy and semantic similarity to the target.

## Context Evaluation

To ensure the accuracy and relevance of the retrieved contexts in the ChatBot responses, I use the Entity Recall metric.

### Entity Recall
Entity Recall measures the system's ability to correctly recall all relevant entities within the context compared to a set of reference entities. This metric evaluates whether the system can identify and retrieve all key entities necessary for a comprehensive understanding of the query context.

**Methodology:**
1. **Extract Entities:** Use Named Entity Recognition (NER) tools such as SpaCy and NLTK to dynamically extract entities from both the retrieved context and the generated answer.
2. **Compare Entities:** Compare the extracted entities from the retrieved context with those from the generated answer.
3. **Calculate Entity Recall:** Calculate the recall as the number of correctly recalled entities divided by the total number of relevant entities in the generated answer.