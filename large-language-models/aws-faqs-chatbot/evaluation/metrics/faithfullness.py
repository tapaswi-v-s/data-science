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

class Faithfullness:

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_new_tokens=1024,
        temperature=1,
        model_kwargs={'token': os.getenv("HUGGINGFACEHUB_API_TOKEN")}
    )

    FAITHFULLNESS_PROMPT = """###Task Description:
    You will be given a question that was sent to a machine learning model, and you will be given a response that the model produced.
    You will also be given a context that was used by the model to generate the response.

    Your task is to determine a numerical metric called faithfullness based on the response and the context.
    A definition of faithfullness is provided below.

    1. Write a detailed feedback that assess the faithfullness of the response with the context strictly based on the given score rubric, not evaluating in general.
    2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
    3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
    4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

    Metric Definition:
    Faithfulness is only evaluated with the provided response and provided context, please ignore the provided 
    input entirely when scoring faithfulness. Faithfulness assesses how much of the provided response is 
    factually consistent with the provided context. A higher score indicates that a higher proportion of 
    claims present in the response can be derived from the provided context. Faithfulness does not 
    consider how much extra information from the context is not present in the response.

    ###The question:
    {question}

    ###Response to evaluate:
    {response}

    ###Context:
    {context}

    ###Score Rubrics:
    [Is the response faithfull to the given context?]
    Score 1: None of the claims in the response can be inferred from the provided context.
    Score 2: Some of the claims in the response can be inferred from the provided context, but the majority of the response is missing from, inconsistent with, or contradictory to the provided context.
    Score 3: Half or more of the claims in the response can be inferred from the provided context.
    Score 4: Most of the claims in the response can be inferred from the provided context, with very little information that is not directly supported by the provided context.
    Score 5: All of the claims in the response are directly supported by the provided context, demonstrating high faithfulness to the provided context.

    Your Output: 
"""

    evaluation_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a fair evaluator language model."),
            HumanMessagePromptTemplate.from_template(FAITHFULLNESS_PROMPT),
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
                'faithfullness_score': score,
                'faithfullness_score_feedback': feedback,
                'question': row[1]['question'],
                'context': row[1]['context'],
                'generated_answer': row[1]['generated_answer']                
            }
            outputs.append(result)

        faithfullness_df = pd.DataFrame(outputs)
        return faithfullness_df