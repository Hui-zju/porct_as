import os
import json
import httpx
import copy
import pandas as pd
import openai
from gen_response import gen_chat_response_with_backoff
from openai import OpenAI
from process_response import get_label_from_response
from norms import system_task, usr_task
from norms import gen_json_format_prompt


def dialogue(text, criteria, client, model, mode="single", temperature=0.2, return_confidence=False):  # , save_path
    criteria_list = ["- " + key + ": " + value for key, value in criteria.items()]  #
    if mode == "single1":
        criteria_prompt = [usr_task + "\n\n" + "# Criteria\n" + "\n".join(criteria_list)]
        format_prompt = [gen_json_format_prompt("all")]
    elif mode == "single2":
        criteria_prompt = [usr_task + "\n\n" + "# Criteria\n" + "\n".join(criteria_list)]
        format_prompt = [gen_json_format_prompt([key for key, value in criteria.items()])]
    elif mode == "gradual":
        gradual_criteria = []
        for i in range(len(criteria_list)):
            gradual_criteria.append("\n".join(criteria_list[:i + 1]))
        criteria_prompt = [usr_task + " ".join(gradual_criteria)]
        format_prompt = [gen_json_format_prompt("i") for i in range(len(criteria_list))]
    else:
        criteria_prompt = [usr_task + "\n\n" + "# Criteria\n" + value for value in criteria_list]
        format_prompt = [gen_json_format_prompt(key) for key, value in criteria.items()]

    task_prompt = system_task

    article_prompt = "# Article\n" + text
    pred_label = [{"response": "", "message": []} for _ in range(len(criteria_prompt))]  # [{"response": ""}] * len(norms)  not value copy
    message = [{"role": "system", "content": task_prompt + "\n\n" + article_prompt}]
    for idy, norm in enumerate(criteria_prompt):
        input_prompt = norm + "\n\n" + format_prompt[idy] + "\n\n" + "ANSWER:\n"
        if mode == "continuous" or mode == "gradual":
            message.append({"role": "user", "content": input_prompt})
        else:
            message = [
                {"role": "system", "content": task_prompt + "\n\n" + article_prompt},
                {"role": "user", "content": input_prompt}
            ]
        while True:
            response_label = gen_chat_response_with_backoff(messages=message, client=client,
                                                            model_chat_engine=model,
                                                            return_confidence=return_confidence,
                                                            temperature=temperature)
            try:
                get_label_from_response(response_label)
                break
            except:
                print("Large model answers do not meet the requirements")
        pred_label[idy]["response"] = response_label
        pred_label[idy]["message"] = copy.deepcopy(message)
        message.append({"role": "assistant", "content": response_label})
        if get_label_from_response(response_label) != 1:
            break
        # if "1" not in response_label:
        #     break
    return pred_label


def dialogues(texts, dialogue_modes, criteria, save_folder, file_prex, client, model, temperature):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for item in dialogue_modes:  #
        out_name = "_".join([file_prex, model, str(temperature), item, ".json"])
        out_path = os.path.join(save_folder, out_name)
        responses = []
        for idx, text in enumerate(texts):
            response = dialogue(text, criteria, client, model, mode=item, temperature=temperature)
            print(f"the {item} model of {idx}th article finished")
            responses.append(response)
            #  Save every 50 times in case of breaking
            if (idx + 1) % 1 == 0 or (idx + 1) == len(texts):
                with open(out_path, 'w') as file:
                    json.dump(responses, file)


def classification_on_txt_folder(dialogue_modes, criteria_dict, data_dir, save_folder, client, model, temperature):
    texts = [open(os.path.join(data_dir, file_name)).read().replace("\n", " ") for file_name in os.listdir(data_dir)]
    dialogues(texts, dialogue_modes, criteria_dict, save_folder, client, model, temperature)


def classification_on_file(dialogue_modes, criteria_dict, data_dir, save_folder, client, model, temperature):
    file_name = os.path.basename(data_dir)
    inc_criteria_names = file_name[:-4].split('+')
    inc_criteria = {key: criteria_dict[key] for key in inc_criteria_names if key in criteria_dict}
    sen_data = pd.read_csv(data_dir).to_dict("records")
    texts = [sen['sentence'].replace("\n", " ") for sen in sen_data]
    save_folder = os.path.join(save_folder, file_name[:-4])
    dialogues(texts, dialogue_modes, inc_criteria, save_folder, client, model, temperature)


def classification_on_folder(dialogue_modes, criteria_dict, data_dir, save_folder, client, model, temperature):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_dir, file_name)
            classification_on_file(dialogue_modes, criteria_dict, file_path, save_folder, client, model, temperature)


if __name__ == '__main__':
    rct_data_dir = "../dataset/api_dataset/prompt_optimization_tuning_set_100.csv"
    sen_data = pd.read_csv(rct_data_dir).to_dict("records")
    texts = [sen['sentence'].replace("\n", " ") for sen in sen_data]

    temperature = 0
    dialogue_modes = ["single1"]

    openai.api_key = "your-api-key"
    client = OpenAI()
    model = "gpt-3.5-turbo"  # "gpt-4o"
    save_folder_path = "../result/api/initial_prompt"
    save_file_prex = os.path.basename(rct_data_dir)[:-4]

    inc_criteria_names = ["format", "population", "purpose", "characteristic"]
    from norms import rct_criteria as criteria_dict
    inc_criteria = {key: criteria_dict[key] for key in inc_criteria_names if key in criteria_dict}
    dialogues(texts, dialogue_modes, inc_criteria, save_folder_path, save_file_prex, client, model, temperature)






