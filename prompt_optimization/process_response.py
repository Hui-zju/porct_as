import json
import httpx
import json_repair
from openai import OpenAI
from json_repair import repair_json
from gen_response import gen_chat_response_with_backoff


def repair_json_by_chatgpt(response):
    client = OpenAI(base_url="https://apikey.gpt12345.top/v1",  # /chat/completions
                    api_key="sk-JbZSVPpC89RFG9FoC0247cE311A54c9b97C439B39b73Cc27",
                    http_client=httpx.Client(
                        base_url="https://apikey.gpt12345.top/v1",
                        follow_redirects=True)
                    )
    model = "gpt-3.5-turbo"
    prompt = "Please ensure that the following JSON output is in correct JSON format without any escaped characters, and please only provide the correct json without any other text in your response." + "\n" + response

    message = [
        {"role": "system", "content": "You are a helpful professor and scientist."},
        {"role": "user", "content": prompt}
    ]
    response_label = gen_chat_response_with_backoff(messages=message, client=client, model_chat_engine=model)
    return response_label


def remove_quotes(s):
    while True:
        stripped_s = s.strip('\'"')
        if stripped_s == s:
            break
        s = stripped_s
    return s


def get_label_from_response(response):
    replace_response = response.replace("\\n", "").replace("\\", "")  #
    remove_response = remove_quotes(replace_response)
    repair_response = repair_json(remove_response)  # , skip_json_loads=True
    repair_response = remove_quotes(repair_response)
    repair_response = eval(repair_response)
    if isinstance(repair_response, dict):
        labels = []
        for key in repair_response.keys():
            if "whether_" in key:
                value = repair_response[key]
                mapping = {"Yes": 1, "No": 0}  # , "Uncertain": -1
                label = mapping.get(value, -1)
                labels.append(label)
        if len(labels) > 0:
            pred_label = 1 if all(element == 1 for element in labels) else 0
            return pred_label
        else:
            return -2
    return -2


