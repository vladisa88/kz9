import sys
# sys.path.append('/home/askatsevalov/repos/kz9-vlad/server/Neural/env/lib/python3.8/site-packages')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import warnings
warnings.filterwarnings('ignore')

def all_work(req_group_link):
    import requests, json, time

    group_link = req_group_link.split('/')[-1] #'dubki'
    num_comments = '100'
    num_posts = '10'
    # api_token = 'e8123e94ad03b22aafce171c3ac3c25e1febd83229f0c17dbe46583de5367cafff2509b57b7a4a4963660' Это токен Владьев
    api_token = 'a4537b3ec7d1869d56a020c630e46559932c2f1df1f4921bb3747690b9de32f41cbb4f1fbf0b08465bdd3' # Это токен Каца
    sample_request = 'https://api.vk.com/method/'
    get_group_by_link = f'utils.resolveScreenName?v=5.53&access_token={api_token}&'
    get_wall_entries = f'wall.get?v=5.53&access_token={api_token}&'
    get_entry_comments = f'wall.getComments?v=5.53&access_token={api_token}&'
    # Получаем группу
    group_id = requests.get(sample_request+get_group_by_link+'screen_name='+group_link).text
    group_id = str(json.loads(group_id)['response']['object_id'])

    # Получаем список последних постов
    entries_respose = requests.get(sample_request+get_wall_entries+
                                'count='+num_posts+'&filter=all&owner_id=-'+group_id).text
    post_ids = [str(x['id']) for x in json.loads(entries_respose)['response']['items']]

    time.sleep(1)
    # Получем сами комменты
    comment_list = ['тест']
    for entry_id in post_ids:
        entry_response = requests.get(sample_request+get_entry_comments+'count='+num_comments+'&owner_id=-'+
                                    group_id+'&post_id='+entry_id).text
        entry_texts = [x['text'] for x in json.loads(entry_response)['response']['items']]
        comment_list.extend(entry_texts)
        time.sleep(0.15)


    ### Подключаем либы и загружаем нейросеть
    import torch
    from transformers import BertForSequenceClassification
    import pandas as pd
    import numpy as np
    from transformers import BertTokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
    
    # Load the BERT tokenizer.
    # print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    MAX_LEN = 200

    model = BertForSequenceClassification.from_pretrained('model_4epochs')
    device = "cpu:0"
    model = model.to(device)
    ###

    # Load the dataset into a pandas dataframe.
    df = pd.DataFrame({'sentence':comment_list,
                    'label':np.zeros(len(comment_list))})

    # Create sentence and label lists
    sentences = df.sentence.values
    labels = df.label.values
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    )
        
        input_ids.append(encoded_sent)
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, 
                            dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask) 
    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)
    # Set the batch size.  
    batch_size = 32  
    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


    # Prediction on test set
    # print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
    # Put model in evaluation mode
    model.eval()
    # Tracking variables 
    predictions , true_labels = [], []
    # Predict 
    for batch in prediction_dataloader:
    # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
    
    # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
    
    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
        with torch.no_grad():
        # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                        attention_mask=b_input_mask)
        logits = outputs[0]
    # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
    
    # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)


    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1,1)

    resulting_toxicity = round(np.mean(softmax(predictions[0])[:,1]), 4)
    if resulting_toxicity <=0.25:
        print(f'Аудитория настроена положительно\n Токсичность аудитории - {resulting_toxicity*100}%')
    elif 0.25 < resulting_toxicity <=0.5:
        print(f'Аудитория настроена нейтрально\n Токсичность аудитории - {resulting_toxicity*100}%')
    elif 0.5 < resulting_toxicity <=0.75:
        print(f'Аудитория настроена немного негативно\n Токсичность аудитории - {resulting_toxicity*100}%')
    elif 0.75 < resulting_toxicity <=0.9:
        print(f'Аудитория настроена негативно\n Токсичность аудитории - {resulting_toxicity*100}%')
    else:
        print(f'Аудитория настроена крайне негативно\n Токсичность аудитории - {resulting_toxicity*100}%')
    

if __name__== "__main__":
    all_work(str(sys.argv[1]))