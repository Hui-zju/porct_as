system_task = "As an expert annotator with a focus on scientific article content analysis, your role involves ascertaining whether the article meets the criteria."
usr_task = "Please ascertain whether the article meets all of the following criteria."  #  to confirm that it is a precision oncology randomised controlled trial

rct_criteria = {
    "format": "The article is a randomised controlled trial that includes a description of the study design, randomisation process, participant information, outcome measures, and specific results.", # The article must provide a description of the RCT design, including randomization methods, control group setup, etc.
    "population": "The study subjects of the article are patients diagnosed with cancer.",
    "purpose": "The article reports specific outcomes of cancer treatment evaluation, including progression-free survival, overall survival, response rates, and more.",  #  Note: Articles with poor results also meet the criterion.
    "characteristic": "The article involves genetic or genomic characteristics, including gene mutation, oligonucleotide variation, copy number variation, gene expression level, gene fusion, chromosome structure variation, epigenetic modification and microsatellite instability.",
    # "signature": "The treatment strategies of the article is based on the DNA signature of patient's tumor, such as EGFR mutation, CD30-positive, HR+, and HER2-." #  related to patient's genetic or genomic characteristics
    }


def gen_json_format_prompt(criteria):
    # format_prompt = f'Provide your response in JSON format, as follows:{{"whether_meet_{criteria}_criteria": "Yes/No", "confidence": "Provide a single confidence percentage for your choice of Yes/No options.", "justification": "Provide a brief justification for your choice of Yes/No options (No more than 50 words)."}}\n\nOptions:\n- Yes\n- No'
    if isinstance(criteria, str):
        criteria = [criteria]
    elif isinstance(criteria, list):
        criteria = criteria
    else:
        return ValueError

    format_prompt = "# Response\nProvide your response in JSON format, as follows:\n"
    option_prompt = "# Options\n- Yes\n- No"
    dict_list = []
    dict_list.append(f'"justification": "Provide a brief analysis if the article meets each criterion."')
    for item in criteria:
        dict_list.append(f'"whether_meet_{item}_criteria": "Yes/No"')
    # dict_list.append(f'"justification": "Provide a brief justification for your choice of Yes/No options."')
    # format_prompt += '"{' + ", ".join(dict_list) + '}"' + "\n\n" + option_prompt

    format_prompt += '{' + ", ".join(dict_list) + '}' + "\n\n" + option_prompt
    return format_prompt