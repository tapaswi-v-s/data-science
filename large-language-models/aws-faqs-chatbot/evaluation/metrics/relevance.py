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

class Relevance:

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_new_tokens=1024,
        temperature=1,
        model_kwargs={'token': os.getenv("HUGGINGFACEHUB_API_TOKEN")}
    )

    RELEVANCE_PROMPT = """###Task Description:
    You will be given a question that was sent to a machine learning model, and you will be given a response that the model produced.
    You will also be given a context that was used by the model to generate the response.

    Your task is to determine a numerical metric called relevance based on the response and the context.
    A definition of faithfullness is provided below.

    1. Write a detailed feedback that assess the relevance of the response with the context strictly based on the given score rubric, not evaluating in general.
    2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
    3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
    4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

    Metric Definition:
    Relevance encompasses the appropriateness, significance, and applicability of the response with respect 
    to both the question and the context. Scores should reflect the extent to which the output directly addresses 
    the question provided in the input, given the provided context.

    ###The question:
    {question}

    ###Response to evaluate:
    {response}

    ###Context:
    {context}

    ###Score Rubrics:
    [Is the response relevant to the given question and the given context?]
    Score 1: The response doesn't mention anything about the question or is completely irrelevant to the provided context.
    Score 2: The response provides some relevance to the question and is somehow related to the provided context.
    Score 3: The response mostly answers the question and is largely consistent with the provided context.
    Score 4: The response answers the question and is consistent with the provided context.
    Score 5: The response answers the question comprehensively using the provided context.

    Your Output: 
"""

    evaluation_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a fair evaluator language model."),
            HumanMessagePromptTemplate.from_template(RELEVANCE_PROMPT),
        ]
    )

    def evaluate(self, answers: pd.DataFrame) -> pd.DataFrame:
        outputs = []
        for row in tqdm(answers.iterrows()):
            prompt = self.evaluation_prompt_template.format_messages(
                question = row[1]['question'],
                context = row[1]['context'],
                response = row[1]['generated_answer'])
            
            eval_result = self.llm.invoke(prompt)

            feedback, score = [item.strip() for item in eval_result.split("[RESULT]")[:2]]
            score = score.strip()[0]

            result = {
                'relevance_score': score,
                'relevance_score_feedback': feedback,
                'question': row[1]['question'],
                'context': row[1]['context'],
                'generated_answer': row[1]['generated_answer']                
            }
            outputs.append(result)

        relevance_df = pd.DataFrame(outputs)
        return relevance_df