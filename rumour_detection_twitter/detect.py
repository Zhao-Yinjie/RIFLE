# from create_inputs import generate_structure
from sgnlp.models.rumour_detection_twitter import (
    RumourDetectionTwitterConfig,
    RumourDetectionTwitterModel,
    RumourDetectionTwitterTokenizer,
    download_tokenizer_files_from_azure,
)
import torch
from torch.nn.functional import softmax


model_card_path="model_card/rumour.json"

config = RumourDetectionTwitterConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_detection_twitter/config.json"
)
model = RumourDetectionTwitterModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/rumour_detection_twitter/pytorch_model.bin",
    config=config,
)
download_tokenizer_files_from_azure(
    "https://storage.googleapis.com/sgnlp/models/rumour_detection_twitter/",
    "rumour_tokenizer",
)
tokenizer = RumourDetectionTwitterTokenizer.from_pretrained("rumour_tokenizer")

id_to_string = {
    0: "a false rumour",
    1: "a true rumour",
    2: "an unverified statement",
    3: "a non-rumour"
}

def generate_structure(thread_len, max_posts):
    time_delay_ids = [0] * thread_len + [1] * (max_posts - thread_len)

    structure_ids = [
        [3] * idx + [4] + [2] * (thread_len - 1 - idx) + [5] * (max_posts - thread_len)
        for idx in range(thread_len)
    ] + [[5] * max_posts] * (max_posts - thread_len)

    post_attention_mask = [1] * thread_len + [0] * (max_posts - thread_len)

    return [time_delay_ids], [structure_ids], [post_attention_mask]

class RumourDetection():

    def predict(self, txt):
        tweet_lst = [txt]
        thread_len = len(tweet_lst)

        token_ids, token_attention_mask = tokenizer.tokenize_threads(
                [tweet_lst],
                max_length=config.max_length,
                max_posts=config.max_tweets,
                truncation=True,
                padding="max_length",
            )

        time_delay_ids, structure_ids, post_attention_mask = generate_structure(
            thread_len=thread_len, max_posts=config.max_tweets
        )

        token_ids = torch.LongTensor(token_ids)
        token_attention_mask = torch.Tensor(token_attention_mask)
        time_delay_ids = torch.LongTensor(time_delay_ids)
        post_attention_mask = torch.Tensor(post_attention_mask)
        structure_ids = torch.LongTensor(structure_ids)

        logits = model(
                token_ids=token_ids,
                time_delay_ids=time_delay_ids,
                structure_ids=structure_ids,
                token_attention_mask=token_attention_mask,
                post_attention_mask=post_attention_mask,
            ).logits

        # Convert the outputs into the format the frontend accepts
        probabilities = softmax(logits, dim=1)
        print(probabilities)
        predicted_y = torch.argmax(logits, dim=1)[0].item()

        return id_to_string[predicted_y]
    
# test
# rumour_detection = RumourDetection()
# out = rumour_detection.predict("US President Donald Trump has tested positive for coronavirus")
# print(out)