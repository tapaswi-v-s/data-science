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

class AnswerCorrectness:

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_new_tokens=1024,
        temperature=1,
        model_kwargs={'token': os.getenv("HUGGINGFACEHUB_API_TOKEN")}
    )

    ANSWER_CORRECTNESS_PROMPT = """###Task Description:
    You will be given a question that was sent to a machine learning model, and you will be given a response that the model produced.
    You will also be given a context that was used by the model to generate the response.

    Your task is to determine a numerical metric called answer_correctness based on the response, the question and the context.
    A definition of faithfullness is provided below.

    1. Write a detailed feedback that assess the correctness of the response with the context and the question strictly based on the given score rubric, not evaluating in general.
    2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
    3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
    4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

    Metric Definition:
    Answer correctness is evaluated on the accuracy of the provided response based on 
    the provided target, which is the ground truth. Scores can be assigned based on 
    the degree of semantic similarity and factual correctness of the provided response 
    to the provided target, where a higher score indicates higher degree of accuracy.

    ###The question:
    {question}

    ###Response to evaluate:
    {response}

    ###Target:
    {target}

    ###Score Rubrics:
    [Is the response relevant to the given question and the given context?]
    Score 1: The response is completely incorrect. It is completely different from or contradicts the provided target.
    Score 2: The response demonstrates some degree of semantic similarity and includes partially correct information. However, the response still has significant discrepancies with the provided target or inaccuracies."
    Score 3: The response addresses a couple of aspects of the input accurately, aligning with the provided target. However, there are still omissions or minor inaccuracies.
    Score 4: The response is mostly correct. It provides mostly accurate information, but there may be one or more minor omissions or inaccuracies.
    Score 5: The response is correct. It demonstrates a high degree of accuracy and semantic similarity to the target.

    Your Output: 
"""

    evaluation_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a fair evaluator language model."),
            HumanMessagePromptTemplate.from_template(ANSWER_CORRECTNESS_PROMPT),
        ]
    )

    def evaluate(self, answers: pd.DataFrame) -> pd.DataFrame:
        outputs = []
        for row in tqdm(answers.iterrows()):
            prompt = self.evaluation_prompt_template.format_messages(
                question = row[1]['question'],
                target = row[1]['context'],
                response = row[1]['generated_answer'])
            
            eval_result = self.llm.invoke(prompt)
            
            feedback, score = [item.strip() for item in eval_result.split("[RESULT]")[:2]]
            score = score.strip()[0]

            result = {
                'correctness_score': score,
                'correctness_score_feedback': feedback,
                'question': row[1]['question'],
                'context': row[1]['context'],
                'generated_answer': row[1]['generated_answer']                
            }
            outputs.append(result)

        correctness_df = pd.DataFrame(outputs)
        return correctness_df