from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
import pandas as pd
from tqdm.auto import tqdm
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.language_models.llms import LLM
import os
from dotenv import load_dotenv
load_dotenv()

class AnswerSimilarity:

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_new_tokens=1024,
        temperature=1,
        model_kwargs={'token': os.getenv("HUGGINGFACEHUB_API_TOKEN")}
    )

    SIMILARITY_EVALUATION_PROMPT = """###Task Description:
    You will be given a question that was sent to a machine learning model, and you will be given a response that the model produced.
    You will also be given a reference answer that was used by the model to generate the response and has a score of 5.

    Your task is to determine a numerical metric called answer_similarity based on the response and the reference answer.
    A definition of answer_similarity is provided below.

    1. Write a detailed feedback that assess the similarity of the response with the reference answer strictly based on the given score rubric, not evaluating in general.
    2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
    3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
    4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

    Metric Definition:
    Answer similarity is evaluated on the degree of semantic similarity of the provided output to the provided targets, 
    which is the ground truth. Scores can be assigned based on the gradual similarity in meaning and description to the 
    provided targets, where a higher score indicates greater alignment between the provided output and provided targets.


    ###The question:
    {question}

    ###Response to evaluate:
    {response}

    ###Reference Answers (Score 5):
    {reference_answer}

    ###Score Rubrics:
    [Is the response semantically similar based on the reference answer?]
    Score 1: The response has little to no semantic similarity to the reference answer.
    Score 2: The response displays partial semantic similarity to the reference answer on some aspects.
    Score 3: The response has moderate semantic similarity to the reference answer.
    Score 4: The response aligns with the reference answer in most aspects and has substantial semantic similarity.
    Score 5: The response closely aligns with the reference answer in all significant aspects.

    Your Output: 
"""

    evaluation_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a fair evaluator language model."),
            HumanMessagePromptTemplate.from_template(SIMILARITY_EVALUATION_PROMPT),
        ]
    )

    def evaluate(self, answers: pd.DataFrame) -> pd.DataFrame:
        outputs = []
        for row in tqdm(answers.iterrows()):
            prompt = self.evaluation_prompt_template.format_messages(
                question = row[1]['question'],
                reference_answer = row[1]['context'],
                response = row[1]['generated_answer'])
            
            eval_result = self.llm.invoke(prompt)
            
            feedback, score = [item.strip() for item in eval_result.split("[RESULT]")[:2]]
            score = score.strip()[0]

            result = {
                'similarity_score': score,
                'similarity_score_feedback': feedback,
                'question': row[1]['question'],
                'context': row[1]['context'],
                'generated_answer': row[1]['generated_answer']                
            }
            outputs.append(result)

        similarity_df = pd.DataFrame(outputs)
        return similarity_df