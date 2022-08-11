"""
We use davinci-instruct-beta-v2, a model specialized in following instructions, to create questions based on the given context. Then we also use davinci-instruct-beta-v2 to answer those questions, given the same context.

This is expensive, and will also take a long time, as we call the davinci engine for each section. You can simply download the final dataset instead.

We're using the dataset created using the previous py

"""

## 2.0 Retrieve model

import os
import openai
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = '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'
openai.Model.retrieve("ada")


## 2.1 Read in the data, and create a context
import pandas as pd

df = pd.read_csv('hpv_data/hpv_sections.csv')
df['context'] = df.title + "\n" + df.heading + "\n\n" + df.content
print (df.head(5))

## 2.2 Create questions based on the context
"""
Use davinci-instruct to generate a number of plausible questions relating to the Wikipedia section contents.

Note: We have used temperature=0, but it may be beneficial to experiment with a higher temperature to get a higher diversity of questions.

**WARNING: This step will last a long time, and consume a lot of tokens, as it calls davinci-instruct for every section to generate a number of questions.**
"""
import openai

def get_questions(context):
    try:
        response = openai.Completion.create(
            engine="davinci-instruct-beta-v2",
            prompt=f"Write questions based on the text below\n\nText: {context}\n\nQuestions:\n1.",
            temperature=0,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response['choices'][0]['text']
    except:
        return ""


df['questions']= df.context.apply(get_questions)
df['questions'] = "1." + df.questions
print(df[['questions']].values[0][0])

print(df.content.values[0])

## 2.3 Create answers based on the context
"""
Use davinci-instruct to answer the questions given the relevant Wikipedia section contents

Note: We have used temperature=0, but it may be beneficial to experiment with a higher temperature to get a higher diversity of questions.

**WARNING: This step will last a long time, and consume a lot of tokens, as it calls davinci-instruct for every section to answer all the questions.**
"""

def get_answers(row):
    try:
        response = openai.Completion.create(
            engine="davinci-instruct-beta-v2",
            prompt=f"Write questions based on the text below\n\nText: {row.context}\n\nQuestions:\n{row.questions}\n\nAnswers:\n1.",
            temperature=0,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['text']
    except Exception as e:
        print (e)
        return ""


df['answers']= df.apply(get_answers, axis=1)
df['answers'] = "1." + df.answers
df = df.dropna().reset_index().drop('index',axis=1)
print(df[['answers']].values[0][0])

## 2.4 Save the HPV Q&A dataset based on Wikipedia sections
df.to_csv('hpv_data/hpv_qa.csv', index=False)


## 2.5 Search file
"""
We create a search file (API reference), which can be used to retrieve the relevant context when a question is asked.
"""
df = df[df.tokens<2000]
df[['context', 'tokens']].rename(columns={'context':'text','tokens':'metadata'}).to_json('hpv_data/hpv_search.jsonl', orient='records', lines=True)

search_file = openai.File.create(
  file=open("hpv_data/hpv_search.jsonl"),
  purpose='search'
)
hpv_search_fileid = search_file['id']

## 2.6 Answer questions based on the context provided
"""
We will use a simple implementation of the answers endpoint. 
This works by simply using the /search endpoint, which searches over an indexed file to 
obtain the relevant sections which can be included in the context, 
following by a question and answering prompt given a specified model.
"""

from answers_with_ft import create_context, answer_question

print("", "answer with finetune", create_context("Is HPV vaccine safe?", hpv_search_fileid, max_len=400))

print("", "davinci-instruct-beta-v2", answer_question(hpv_search_fileid, "davinci-instruct-beta-v2", 
            "Is HPV vaccine safe?"))

print("", "davinci-instruct-beta-v2 with max_len param", answer_question(hpv_search_fileid, "davinci-instruct-beta-v2", 
            "Is HPV vaccine safe?", max_len=1000))


## 2.7 (Optional) Investigation into how likely the search endpoint is to return the relevant context
def check_context(title, heading, question, max_len=1800, search_model='ada', max_rerank=10):
    try:
        results = openai.Engine(search_model).search(
            search_model=search_model, 
            query=question, 
            max_rerank=max_rerank,
            file=hpv_search_fileid,
            return_metadata=True
        )
        index=-1
        returns = []
        cur_len = 0
        for result in results['data']:
            cur_len += int(result['metadata']) + 4 # we add 4 tokens for the separator `\n\n###\n\n`
            if cur_len > max_len:
                break
            returns.append(result['text'])
            res = result['text'].split('\n')
            if res[0] == title and res[1] == heading:
                index = len(returns) - 1
                break
        return index, cur_len
    except Exception as e:
        #print (e)
        return []
print("\n 2.7 (Optional) Investigation into how likely the search endpoint is to return the relevant context \n", 
        check_context("HPV safety", "Vaccines", "Why do we need to have HPV vaccines", max_len=10000))

"""
We utilize the generated questions based on context to estimate how often we can retrieve the original context. These questions are noisy, so this is not a perfect estimate.

Our questions and answers are prefixed with numbered bullet points, however due to the way they were generated, they are missing the first number, hence we add "1." to the list of questions (and answers).

We calculate the rank of the section retrieved using ada search, and the number of tokens in the context needed to retrieve the relevant section in full.
"""

ada_results = df.apply(lambda x: [
                    check_context( x.title, 
                                   x.heading, 
                                   q[3:],     # remove the number prefix
                                   max_len=1000000, # set a large number to get the full context 
                                   model='ada', 
                                   max_rerank=200,
                                 ) 
                    for q in (x.questions).split('\n') # split the questions
                    if len(q) >10 # remove the empty questions
                ], axis=1)
print ('\n ada results. \n', ada_results.head())

out = pd.concat([ada_results], axis=1)
out.columns = ['ada']
out.to_csv('hpv_data/search_engine_results.csv')

def expand_lists(out):
    """
    Expand a pandas series containing lists into a series, where each list element becomes a value on its own

    Input is a row per paragraph, which has multiple questions
    Output is a row per question
    """
    cols = [pd.DataFrame(out[name].tolist()).stack().reset_index(level=1, drop=True).rename(name) for name in out.columns] 
    return pd.concat(cols, axis=1)

out_expanded = expand_lists(out)
out_expanded['rank'] = out_expanded.ada.apply(lambda x: x[0] if x != [] else -2)
out_expanded['tokens'] = out_expanded.ada.apply(lambda x: x[1] if x != [] else -2)

within_2k = (out_expanded.tokens < 2000).mean()
print(f"{within_2k*100:.1f}% of relevant paragraphs are retrieved within the first 2k tokens")

outside_200 = (out_expanded['rank'] == -1).mean()
print(f"{outside_200*100:.1f}% of relevant paragraphs are not retrieved within the first 200 results")

import matplotlib.pyplot as plt

# plot a histogram, and add axis descriptions and title
out_expanded[(out_expanded['rank'] >=0)&(out_expanded['rank'] <30)]['rank'].hist(bins=29)
plt.xlabel('rank')
plt.ylabel('count')
plt.title('Histogram of ranks of retrieved paragraphs')
plt.show()

out_expanded[(out_expanded.tokens>=0)&(out_expanded.tokens < 2000)]['tokens'].hist(bins=29)
plt.xlabel('tokens')
plt.ylabel('count')
plt.title('Histogram of the number of minimum tokens needed')
plt.show()

# normalized value_counts
print(out_expanded['rank'].value_counts(normalize=True).sort_index()[:13])