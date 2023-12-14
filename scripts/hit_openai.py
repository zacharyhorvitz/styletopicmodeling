import openai
import argparse
import json
import os
import random
from tqdm import tqdm
from datetime import datetime
import time

import sys

from openai import OpenAI

client = OpenAI()



def get_author_data(*, author_name, author_directory, shard='train'):
    clean_name = author_name.replace(":","").replace(" ", "_")
    with open(os.path.join(author_directory, clean_name, f'{shard}.txt'), 'r') as f:
        lines = f.readlines()
        input_data = [line.strip().split('\t') for line in lines]
    return input_data


def hit_openai(samples):

    message = 'Here are some example emails written by the same author: \n'
    for sample in samples:
        message += '{"email body": "' + sample + '"}\n'
    message += "Can you write another short email that could also have been written by the same author, in the same style? Please only include the body of the email, not the subject line or any other metadata."
    print(message)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content

if __name__ == '__main__':

    random.seed(42)

    DATA_SRC = '/burg/nlp/users/zfh2000/enron_processed.json'
    # MAX=10

    with open(DATA_SRC, 'r') as in_file:
        data = json.load(in_file)

    with open('authors_splits.json', 'r') as in_file:
        splits = json.load(in_file)

    train = splits['train']

    sampled_authors = random.sample(train, 100)

    


    authors_to_emails = {a:[] for a in sampled_authors}

    random.shuffle(data)

    for d in data:
        if d['info']['from'] in authors_to_emails:
            authors_to_emails[d['info']['from']].append(d['text'])
    
    # import pdb; pdb.set_trace()

    chat_gpt_out = []

    with open('chat_gpt_out.jsonl', 'w+') as out_file:
        for author in tqdm(authors_to_emails):
            for i in range(5):
                samples = random.sample(authors_to_emails[author], min(3, len(authors_to_emails[author])))

            
                while True:
                    try: 
                        result = hit_openai(samples)
                        break
                    except: #(OpenAI.error.ServiceUnavailableError, OpenAI.error.APIConnectionError, OpenAI.error.APIError, OpenAI.error.Timeout) as e:
                        print("Service unavailable, retrying...")
                        time.sleep(20)
                        continue
                
                print(result)
                chat_gpt_out.append(dict(author=author, samples=samples, result=result))
                out_file.write(json.dumps(dict(author=author, samples=samples, result=result)) + '\n')

    import pdb; pdb.set_trace()





    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--out_dir', type=str)
    # parser.add_argument('--author_directory', type=str)
    # parser.add_argument('--assignments_json', type=str)
    # parser.add_argument('--max_examples', type=int, default=16)
    # parser.add_argument('--approach', type=str)

    # cmd_args = parser.parse_args()
    # hparams = vars(cmd_args)
    # out_dir = hparams['out_dir']
    # approach = hparams['approach']

    # assert approach in ['chatgpt']
    
    # dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # task_folder = f"{out_dir}/{dtime}"
    # os.makedirs(task_folder, exist_ok=False)

    # with open(os.path.join(task_folder, "hparams.json"), 'w') as f:
    #     json.dump(hparams, f)

    # with open(hparams['assignments_json'], 'r') as f:
    #     assignments = json.load(f)

    # total_transfers = sum([len(assignments[source_author]['target'])*len(assignments[source_author]['test_samples']) for source_author in assignments.keys()])


    # counter = -1
    # with open(os.path.join(task_folder, f"style.jsonl"), 'w+') as out: 
    #     with tqdm(total=total_transfers) as pbar:
    #         for source_author in sorted(assignments.keys()):
    #             val_examples = assignments[source_author]['test_samples']
    #             target_authors = assignments[source_author]['target']
    #             for target_author in target_authors:
    #                 target_author_training = [x[1] for x in get_author_data(author_name=target_author, author_directory=hparams['author_directory'], shard='train')]
    #                 for paraphrase, original_text in val_examples:
    #                     # import pdb; pdb.set_trace()
    #                     counter += 1
    #                     if counter < 925: #715 + 121:
    #                         print('skipping', counter)
    #                         continue
    #                     target_texts=target_author_training[:hparams['max_examples']]
    #                     result = do_prompted_transfer(original_text=original_text, examples=target_texts)
    #                     result = dict(
    #                         input_label=source_author,
    #                         paraphrase=paraphrase,
    #                         original_text=original_text,
    #                         target_label=target_author,
    #                         decoded=result)
                    
    #                     # print(f'{original_text} -> {paraphrase} -> {result}')
    #                     out.write(json.dumps(result) + '\n')
    #                     pbar.update(1)

