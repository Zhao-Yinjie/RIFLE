import os
import openai
import pandas as pd


def init_openai_api():
    # Load your API key from an environment variable or secret management service
    openai.api_key = "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"

def get_fs_example(category, input_index, file_type, data_csv_pth): # fs: few shot, get example for few shot
    data_df = pd.read_csv(data_csv_pth)
    examples = data_df['txt'][input_index]
    prompt = file_type + 'examples in the category of ['+category+'] are: \n'
    for i, example in enumerate(examples):
        if i != len(examples)-1:
            prompt += str(i+1) + '. '+ '"'+example+'", \n'
        else:
            prompt += str(i+1) + '. '+'"'+example+'"; \n'
    return prompt

def update_data_csv(csv_pth, new_val, trg_idx):
    data_df = pd.read_csv(csv_pth)
    data_df['category'].iloc[trg_idx]=new_val
    data_df.to_csv(csv_pth, index=False)

def get_target_input(data_csv_pth, trg_idx): # get target input
    data_df = pd.read_csv(data_csv_pth)
    input = data_df['txt'][trg_idx]
    return input

def few_shot(cats, eg_prompt, target_input, file_type): 
    # modify the prompt
    prompt = eg_prompt
    prompt += 'Target'+ file_type+': \n'
    prompt += '"'+str(target_input)+'"; \n'
    prompt += 'Do a clasification of the following '+file_type+'. Available categories are: \n'
    
    for i, cat in enumerate(cats):
        if i != len(cats)-1:
            prompt += '['+cat+'], '
        else:
            prompt += '['+cat+']. '
    
    prompt += 'The target '+file_type+' is in the category of: \n'
    # print(f'Prompt: {prompt}')

    # call openai api and get result
    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=64)

    return response["choices"][0]["text"]

def zero_shot(cats, target_input, file_type):
    # modify prompt
    prompt = '"'+str(target_input)+'"; \n'
    prompt += 'Do a clasification of the following '+file_type+'. Available categories are: \n'
    
    for i, cat in enumerate(cats):
        if i != len(cats)-1:
            prompt += '['+cat+'], '
        else:
            prompt += '['+cat+']. '
    # call openai api and get result
    prompt += 'The target '+file_type+' is in the category of: \n'
    print(f'Prompt: {prompt}')

    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=64)

    return response["choices"][0]["text"]

def change_input_format(input_dict):
    temp = {}
    for key in input_dict:
        value = input_dict[key]
        if key != '' and value != '':
            str_list = value.split(',')
            int_list = [int(x) for x in str_list]
            temp[key] = int_list
    input_dict = temp
    # print(input_dict)
    return input_dict

class Classifier():
    def run(self, mode, file_type, input_dict):
        init_openai_api()
        # define input & categories
        # semantic level classification demo: 
        # 1. Stress vs relaxing
        # 2. School vs career vs talk
        # targets: 95
        email_csv_pth = "email_list.csv"
        video_csv_pth = "video_list.csv"
        type_error = False
        input_dict = change_input_format(input_dict)

        if file_type == 'email':
            data_csv_pth = email_csv_pth
        elif file_type == 'video':
            data_csv_pth = video_csv_pth
        else: 
            print('file type Error')
            type_error = True
        
        print(f'file type {file_type}')
        if type_error == False:
            input_example_dict = input_dict
            cats = [key for key in input_dict]
            print(cats)
            prompt = ''
            for cat, idx in input_example_dict.items():
                prompt += get_fs_example(cat, idx, file_type, data_csv_pth)

            target_csv = pd.read_csv(data_csv_pth)['category'].to_list()
            # total = len(target_csv)
            total = 30
            for i in range(total): # limit to 35 due to api limit
                target_input = get_target_input(data_csv_pth, i)
                print(f'In progress {i} / total {total}')
                out = None
                if mode == 'few_shot':
                    out = few_shot(cats, eg_prompt=prompt, target_input=target_input, file_type=file_type)
                elif mode == 'zero_shot':
                    out = zero_shot(cats, target_input=target_input, file_type=file_type)
                else: 
                    print('Mode should be few_shot or zero_shot')

                update_data_csv(csv_pth=data_csv_pth, new_val=out, trg_idx=i)
        # print(f'Davinci said: {out}')
        # return 

# test
# update_data_csv(csv_pth='email_list.csv', new_val='test', trg_idx=0)