import httpx
import time
import numpy as np


def gen_chat_response_with_backoff(messages, client, model_chat_engine, return_confidence=False, temperature=0.3):
    # prompt_succeeded = False
    # prompt_succeeded = True
    # not prompt_succeeded
    wait_time = 10
    while True:
        try:
            if return_confidence:
                completion = client.chat.completions.create(
                    model=model_chat_engine,
                    messages=messages,
                    temperature=temperature,
                    logprobs=True,
                    # top_logprobs=2,
                )
                response = completion.choices[0].message.content
                logprobs = []
                logprobs_response = completion.choices[0].logprobs.content
                for logprob in logprobs_response:
                    logprobs.append(logprob.logprob)
                conf_score = np.round(np.mean(np.exp(logprobs)), 6)
                return response, conf_score
            else:
                completion = client.chat.completions.create(
                    model=model_chat_engine,
                    messages=messages,
                    temperature=temperature,
                )
                response = completion.choices[0].message.content
                return response
        except:
            print('  LM response failed. Server probably overloaded. Retrying after ', wait_time, ' seconds...')
            time.sleep(wait_time)
            wait_time += wait_time  # exponential backoff
            if wait_time > 3000:
                print(' Finally LM response failed')
                return None


def gen_chat_response(messages, model_chat_engine="gpt-3.5-turbo"):
    completion = client.chat.completions.create(
        model=model_chat_engine,
        messages=messages
    )
    response = completion.choices[0].message.content
    return response


def singe_dialogue(prompt, model_chat_engine):
    messages = [
            {"role": "system",
             "content": "You are a helpful professor and scientist."},
            {"role": "user", "content": prompt}
        ]
    response = gen_chat_response_with_backoff(messages, client, model_chat_engine)
    return response








