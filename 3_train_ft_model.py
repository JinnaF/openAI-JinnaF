"""
3. Train a fine-tuning model specialized for Q&A
https://github.com/openai/openai-python/blob/main/examples/finetuning/olympics-3-train-qa.ipynb
"""

## TODO: change olympics to hpv

import openai
import pandas as pd
df = pd.read_csv('olympics-data/olympics_qa.csv')
olympics_search_fileid = "file-c3shd8wqF3vSCKaukW4Jr1TT"
df.head()

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
len(train_df), len(test_df)


df.context.str.contains('->').sum()

import random

def get_random_similar_contexts(question, context, file_id=olympics_search_fileid, search_model='ada', max_rerank=10):
    """
    Find similar contexts to the given context using the search file
    """
    try:
        results = openai.Engine(search_model).search(
            search_model=search_model, 
            query=question, 
            max_rerank=max_rerank,
            file=file_id
        )
        candidates = []
        for result in results['data'][:3]:
            if result['text'] == context:
                continue
            candidates.append(result['text'])
        random_candidate = random.choice(candidates)
        return random_candidate
    except Exception as e:
        print(e)
        return ""

def create_fine_tuning_dataset(df, discriminator=False, n_negative=1, add_related=False):
    """
    Create a dataset for fine tuning the OpenAI model; either for a discriminator model, 
    or a model specializing in Q&A, where it says if no relevant context is found.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the question, answer and context pairs
    discriminator: bool
        Whether to create a dataset for the discriminator
    n_negative: int
        The number of random negative samples to add (using a random context)
    add_related: bool
        Whether to add the related contexts to the correct context. These are hard negative examples

    Returns
    -------
    pd.DataFrame
        The dataframe containing the prompts and completions, ready for fine-tuning
    """
    rows = []
    for i, row in df.iterrows():
        for q, a in zip(("1." + row.questions).split('\n'), ("1." + row.answers).split('\n')):
            if len(q) >10 and len(a) >10:
                if discriminator:
                    rows.append({"prompt":f"{row.context}\nQuestion: {q[2:].strip()}\n Related:", "completion":f" yes"})
                else:
                    rows.append({"prompt":f"{row.context}\nQuestion: {q[2:].strip()}\nAnswer:", "completion":f" {a[2:].strip()}"})

    for i, row in df.iterrows():
        for q in ("1." + row.questions).split('\n'):
            if len(q) >10:
                for j in range(n_negative + (2 if add_related else 0)):
                    random_context = ""
                    if j == 0 and add_related:
                        # add the related contexts based on originating from the same wikipedia page
                        subset = df[(df.title == row.title) & (df.context != row.context)]
                        
                        if len(subset) < 1:
                            continue
                        random_context = subset.sample(1).iloc[0].context
                    if j == 1 and add_related:
                        # add the related contexts based on the most similar contexts according to the search
                        random_context = get_random_similar_contexts(q[2:].strip(), row.context, search_model='ada', max_rerank=10)
                    else:
                        while True:
                            # add random context, which isn't the correct context
                            random_context = df.sample(1).iloc[0].context
                            if random_context != row.context:
                                break
                    if discriminator:
                        rows.append({"prompt":f"{random_context}\nQuestion: {q[2:].strip()}\n Related:", "completion":f" no"})
                    else:
                        rows.append({"prompt":f"{random_context}\nQuestion: {q[2:].strip()}\nAnswer:", "completion":f" No appropriate context found to answer the question."})

    return pd.DataFrame(rows)

for name, is_disc in [('discriminator', True), ('qa', False)]:
    for train_test, dt in [('train', train_df), ('test', test_df)]:
        ft = create_fine_tuning_dataset(dt, discriminator=is_disc, n_negative=1, add_related=True)
        ft.to_json(f'{name}_{train_test}.jsonl', orient='records', lines=True)

"""
3.2 Submit the datasets for fine-tuning
"""
# !openai api fine_tunes.create -t "olympics-data/discriminator_train.jsonl" -v "olympics-data/discriminator_test.jsonl" --batch_size 16  --compute_classification_metrics --classification_positive_class " yes" --model ada

# !openai api fine_tunes.create -t "olympics-data/qa_train.jsonl" -v "olympics-data/qa_test.jsonl" --batch_size 16

"""
3.3 Using the fine-tuned model
"""
ft_discriminator = "curie:ft-openai-internal-2021-08-23-23-58-57"
ft_qa = "curie:ft-openai-internal-2021-08-23-17-54-10"

def apply_ft_discriminator(context, question, discriminator_model):
    """
    Apply the fine tuned discriminator to a question, to assess whether it can be answered from the context.
    """
    prompt = f"{context}\nQuestion: {question}\n Related:"
    result = openai.Completion.create(model=discriminator_model, prompt=prompt, max_tokens=1, temperature=0, top_p=1, n=1, logprobs=2)
    return result['choices'][0]['logprobs']['top_logprobs']

apply_ft_discriminator('The first human-made object in space was the Soviet Union satellite Sputnik 1 on 4 October 1957.', 
                        'What was the first human-made object in space?', ft_discriminator)

def apply_ft_qa_answer(context, question, answering_model):
    """
    Apply the fine tuned discriminator to a question
    """
    prompt = f"{context}\nQuestion: {question}\nAnswer:"
    result = openai.Completion.create(model=answering_model, prompt=prompt, max_tokens=30, temperature=0, top_p=1, n=1, stop=['.','\n'])
    return result['choices'][0]['text']

apply_ft_qa_answer('The first human-made object in space was the Soviet Union satellite Sputnik 1 on 4 October 1957.', 
                    'What was the first human-made object in space?', ft_qa)

apply_ft_qa_answer('The first human-made object in space was the Soviet Union satellite Sputnik 1 on 4 October 1957.',
                    'What is impressive about the Soviet Union?', ft_qa)


"""
We can see that the model knows when to answer the question, and when to say that insufficient context is present to answer the question.

We can also combine a discriminator and a base model, or a fine-tuned Q&A model. 

Discriminator can essentially serve as a decision whether the question can be answered given the context or not.
"""
def answer_question_conditionally(answering_model, discriminator_model, context, question, discriminator_logprob_yes_modifier=0):
    logprobs = apply_ft_discriminator(context, question, discriminator_model)
    yes_logprob = logprobs[' yes'] if ' yes' in logprobs else -100
    no_logprob = logprobs[' no'] if ' no' in logprobs else -100
    if yes_logprob + discriminator_logprob_yes_modifier < no_logprob:
        return " No appropriate context found to answer the question based on the discriminator."
    return apply_ft_qa_answer(context, question, answering_model)
answer_question_conditionally(ft_qa, ft_discriminator, 
                                "Crowdless games are a rare although not unheard-of occurrence in sports. \
                                 When they do occur, it is usually the result of events beyond the control \
                                 of the teams or fans, such as weather-related concerns, public health concerns, \
                                 or wider civil disturbances unrelated to the game. For instance, \
                                 the COVID-19 pandemic caused many sports leagues around the world \
                                 to be played behind closed doors.",
                                "Could weather cause a sport event to have no crowd?")

from answers_with_ft import answer_question
answer_question(olympics_search_fileid, ft_qa, "Which country won the Women's football tournament at the 2020 Olympic games?")
