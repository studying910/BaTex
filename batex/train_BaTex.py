'''
It is part of the code of paper "Efficient Personalized Text-to-image Generation by Leveraging Textual Subspace".
The paper is submitting to NeurIPS 2023. Please do not distribute.
'''

import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class WeightVector(nn.Module):
    def __init__(self, args):
        super().__init__()
        init_weight = torch.zeros([args.mask_k, 1])
        init_weight[0] = 1.0
        weight_param = nn.Parameter(init_weight, requires_grad=True)
        self.register_parameter("weight", weight_param)

    def forward(self, text_encoder, input_ids, candidate_embedding_matrix, placeholder_token_id):

        output_attentions = text_encoder.text_model.config.output_attentions
        output_hidden_states = text_encoder.text_model.config.output_hidden_states
        return_dict = text_encoder.text_model.config.use_return_dict

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        seq_length = input_ids.shape[-1]
        position_ids = text_encoder.text_model.embeddings.position_ids[:, :seq_length]

        inputs_embeds = text_encoder.text_model.embeddings.token_embedding(input_ids).detach()
        input_ids_numpy = input_ids.squeeze(0).cpu().numpy()
        placeholder_token_index = np.argwhere(input_ids_numpy == placeholder_token_id)[0][0]
        inputs_embeds[:, placeholder_token_index, :] = torch.mm(
            torch.t(self.weight).to(candidate_embedding_matrix.device), candidate_embedding_matrix).squeeze(0)

        position_embeddings = text_encoder.text_model.embeddings.position_embedding(position_ids)
        hidden_states = inputs_embeds + position_embeddings

        bsz, seq_len = input_shape
        causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len,
                                                                                     hidden_states.dtype).to(
            hidden_states.device)

        encoder_outputs = text_encoder.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = text_encoder.text_model.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
        if not return_dict:
            updated_embedding = (last_hidden_state, pooled_output) + encoder_outputs[1:]
        else:
            updated_embedding = BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        return updated_embedding


def part_main():
    mask_k = 576
    placeholder_token = "*"
    initializer_token = "INITIAL_WORD"
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=None
    )
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    text_encoder.resize_token_embeddings(len(tokenizer))

    token_embeds = text_encoder.get_input_embeddings().weight.data
    vocab_embedding = token_embeds[:-1, :]
    vocab_num = len(tokenizer) - 1
    initialization_embedding = token_embeds[initializer_token_id]

    vocab_dist = torch.mm(vocab_embedding, initialization_embedding.unsqueeze(1)).squeeze(1)
    _, vocab_ids = torch.topk(vocab_dist, mask_k, 0, True)
    candidate_embedding_matrix = token_embeds[vocab_ids]
