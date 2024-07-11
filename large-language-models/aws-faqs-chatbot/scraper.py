import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://web.archive.org/web/20240706034001/https://aws.amazon.com/ec2/faqs/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

divs = soup.find_all('div', class_='lb-grid')

def get_text_including_a_and_span_tags(p_tag):
    # Collect text pieces
    text_pieces = []
    for element in p_tag.descendants:
        if element.name in ['a', 'span']:
            text_pieces.append(element.get_text())
        elif isinstance(element, str):
            text_pieces.append(element)
    return ''.join(text_pieces).strip()

current_question = None
current_answer = []
data = []
for div in divs:
    div_children = div.find_all('div', class_='lb-txt-16 lb-rtxt')
    for div1 in div_children:
        ps = div1.find_all('p')
        for p in ps:
            b = p.find('b')
            if b is not None and b.text[0] == 'Q':
                if current_question:
                    data.append({'question': current_question, 'answer': ' '.join(current_answer)})
                    current_answer = []
                current_question = b.text
            else:
                current_answer.append(get_text_including_a_and_span_tags(p))
        
        if current_question:
            data.append({'question': current_question, 'answer': ' '.join(current_answer)})


df = pd.DataFrame(data)
df.to_csv('aws_faqs.csv', index=False)