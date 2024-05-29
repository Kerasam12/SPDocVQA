import random
import numpy as np

import torch
import torch.nn as nn
from click.core import batch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import models._model_utils as model_utils
from models._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
import transformers.models.t5.modeling_t5
import matplotlib.pyplot as plt
import seaborn as sns
from bertviz import model_view



class ProxyVT5:
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])# t5-base  config['model_weights']
        
        
        self.model = T5ForConditionalGeneration.from_pretrained(config['model_weights'], output_attentions=True).to(config['device'])#config['model_weights']
        self.page_retrieval = config['page_retrieval'].lower() if 'page_retrieval' in config else None
        self.max_source_length = config.get('max_source_length', 512)
        t5_config = CustomT5Config.from_pretrained(config['model_weights'])#config['model_weights']
        t5_config.visual_module_config = config['visual_module']

        self.spatial_embedding = SpatialEmbeddings(t5_config).to(config['device'])
        self.visual_embedding = VisualEmbeddings(t5_config).to(config['device'])
        self.tokenizer.add_tokens(['<noans>'])
        self.model.resize_token_embeddings(len(self.tokenizer))


    def parallelize(self):
        self.model = nn.DataParallel(self.model)

    def prepare_inputs_for_vqa(self, question, words, boxes, images, answers=None):
        bs = len(words)
        # input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, words)]
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        prompt_box = [0, 0, 1000, 1000]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0  # To become [0, 0, 0, 0] array.

        # Get input_ids, attention_mask and boxes.
        longest_seq = 0
        batch_input_ids = []
        batch_input_boxes = []
        for batch_idx in range(bs):
            tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
            input_ids = tokenized_prompt.input_ids[:-1]
            input_boxes = [prompt_box] * len(input_ids)

            for word, box in zip(words[batch_idx], boxes[batch_idx]):
                tokenized_word = self.tokenizer(word).input_ids[:-1]  # Tokenize the word and ignore eos_token
                input_ids.extend(tokenized_word)
                input_boxes.extend([box]*len(tokenized_word))  # Repeat the box for each token corresponding to the word.

            batch_input_ids.append(input_ids[:self.max_source_length-1] + [self.tokenizer.eos_token_id])  # Append the eos_token at the end.
            batch_input_boxes.append(np.concatenate([input_boxes[:self.max_source_length-1],  np.array([eos_box])]))  # Append a bounding box corresponding to the eos_token.
            longest_seq = min(max(longest_seq, len(input_ids) + 1), self.max_source_length)
            

        # Convert to tensors and pad. Actually, a pad tensor is created and it's filled with corresponding values.
        tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4],  fill_value=padding_box_value, dtype=torch.long)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)

        for batch_idx in range(bs):
            tensor_input_ids[batch_idx, :len(batch_input_ids[batch_idx])] = torch.LongTensor(batch_input_ids[batch_idx])
            tensor_boxes[batch_idx, :len(batch_input_boxes[batch_idx])] = torch.from_numpy(batch_input_boxes[batch_idx][:len(batch_input_boxes[batch_idx])])
            tensor_attention_mask[batch_idx, :len(batch_input_ids[batch_idx])] = 1

        """
        context = [(' ').join(doc_words) for doc_words in words]
        input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, context)]
        tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        input_embeds = self.model.shared(tokens.input_ids)
        """

        # Send everything to GPU
        tensor_input_ids = tensor_input_ids.to(self.model.device)
        tensor_boxes = tensor_boxes.to(self.model.device)
        tensor_attention_mask = tensor_attention_mask.to(self.model.device)

        # Get semantic and spatial embeddings
        semantic_embedding = self.model.shared(tensor_input_ids)
        spatial_embedding = self.spatial_embedding(tensor_boxes).to(self.model.device)
        visual_embedding, visual_emb_mask = self.visual_embedding(images)

        # input_embeds = semantic_embedding
        input_embeds = torch.add(semantic_embedding, spatial_embedding)
        input_embeds = torch.cat([input_embeds, visual_embedding], dim=1)  # Concatenate semantic + visual embeddings TODO: Provide visual bounding boxes.
        tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1)

        """
        context = [' '.join(doc_words) for doc_words in words]
        input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, context)]
        tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        x = self.model.shared(tokens.input_ids)
        """

        # Tokenize answers
        if answers is not None:
            answers = [random.choice(answer) for answer in answers]
            labels = self.tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
            labels = labels.input_ids.to(self.model.device)
        else:
            labels = None

        return input_embeds, tensor_attention_mask, labels
    


    def forward(self, batch, return_pred_answer=False):
        question = batch['questions']
        words = batch['words']
        boxes = batch['boxes']
        images = batch['images']
        answers = batch['answers']
        bs = len(question)
        pred_answer_pages = []
        if self.page_retrieval == 'logits':
            num_pages = batch['num_pages']
            outputs = []
            pred_answers = []
            pred_answer_pages = []
            pred_answers_conf = []

            for batch_idx in range(bs):
                input_embeds, attention_mask, _ = self.prepare_inputs_for_vqa([question[batch_idx]]*num_pages[batch_idx], words[batch_idx], boxes[batch_idx])  # Answers are not considered. Logits set-up is made only for inference.
                pred_answer, logits, encoder_attention, decoder_attention, cross_attention = self.get_answer_from_model_output(input_embeds, attention_mask)
                # input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip([question[batch_idx]]*len(context[batch_idx]), context[batch_idx])]
                # tokens = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
                #self.visualize_txt_attentions(encoder_attention, decoder_attention, cross_attention, words, pred_answers, question)
                #self.visualize_txt_attentions(encoder_attentions, words, output_tok)
                max_logits = -999999
                answer_page = None
                best_answer = None
                for page_ix in range(len(input_embeds)):
                    if logits[page_ix] > max_logits:
                        max_logits = logits[page_ix]
                        answer_page = page_ix
                        best_answer = pred_answer[page_ix]

                outputs.append(None)  # outputs.append(document_outputs)  # During inference outputs are not used.
                pred_answers.append(best_answer)
                pred_answer_pages.append(answer_page)
                pred_answers_conf.append(max_logits)

        else:
            input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(question, words, boxes, images, answers)
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
            encoder_attentions = outputs.encoder_attentions
            cross_attentions = outputs.cross_attentions
            decoder_attentions = outputs.decoder_attentions
            
            pred_answers, pred_answers_conf, encoder_attention, decoder_attention, cross_attention = self.get_answer_from_model_output(input_embeds, attention_mask) if return_pred_answer else None
            self.visualize_txt_attentions_light(encoder_attentions, decoder_attentions,cross_attentions, words, pred_answers,question, answers)
            #print("\n VISUALIZING ATTENTION \n")
            #self.visualize_txt_attentions(outputs, words, pred_answers, labels)
            if self.page_retrieval == 'oracle':
                pred_answer_pages = batch['answer_page_idx']

            elif self.page_retrieval == 'concat':
                pred_answer_pages = None

        return outputs, pred_answers, pred_answer_pages, pred_answers_conf

    def visualize_txt_attentions_light(self,encoder_attentions, decoder_attentions,cross_attentions, input_tok, output_tok, question,answers):#, input_text, output_text):
        output_ids = self.tokenizer(output_tok)
        print(output_ids)
        output_tokens =  [self.tokenizer.decode(ids) for ids in output_ids.input_ids[-1]]
        tens = 0
        #print("input tokens",type(input_tok), len(input_tok))
        print(input_tok)
        print("output_tok", type(output_tokens), len(output_tokens))
        print(output_tokens)
        print("cross att len:", len(cross_attentions))
        '''for idx,x in enumerate(list(cross_attentions)):
            #tens = torch.stack(decoder_attentions)
            print(x[-1].shape)
            if type(tens) == int:
                tens = x[-1]
            else:
                tens = torch.concat((tens,x[-1]), dim = 2)
            
            #print(x[5].shape)'''
        
        #sum_vals = torch.sum(tens, dim= 2)
        tens = cross_attentions[-1][-1].squeeze()
        print("Tensor Shape", tens.shape)
        
        max_vals, max_idxs = torch.topk(tens.squeeze(), dim = -1, k = 5)   
        
        
        #max_vals = max_vals/tens.shape[2]    
        
        num_layers = len(cross_attentions)
        print(max_idxs[-1])
        print(max_vals[-1])
        #input_tok = [tok for i,tok in enumerate(input_tok[-1]) if i in max_idxs[-1][-1]]
        annot_labels = []
        for words in max_idxs[-1]:
            annot = []
            for i,tok in enumerate(input_tok[-1]):
                if i in list(words):
                    annot.append(str(tok))
            if len(annot) < 5:
                for c in range(5- len(annot)):
                    annot.append("<pad>") 
            
            annot_labels.append(annot)
        
        
        #sp = annot_labels.shape
        #print(np.array(annot_labels))
        
        fig = plt.figure(figsize=(15, 5))
        
        attention = max_vals[-1].squeeze().cpu().numpy()
        #annot_labels = np.asarray(annot_labels).reshape(attention.shape)
        print("annotation_labels:", annot_labels)
        #print("labels:", labels)
        #print("INPUT_TOKENS",input_tok)
        print("OUTPUT TOKENS:",output_tok)
        print("output tok:", output_tokens)
        print("Attentiin", attention.shape)
        if len(output_tokens) < attention.shape[0]:
            for idx in range(( attention.shape[0]- len(output_tokens))):
                output_tokens.append("<pad>")
        ax = plt.axes()
        sns.heatmap(attention, cmap='viridis', cbar=True, annot=annot_labels, yticklabels = output_tokens, fmt = '', ax = ax)#, xticklabels = input_tok
        print(question[-1], answers[-1][0])
        print(output_tokens)
        print("\n", input_tok[-1])
        ax.set_title(question[-1] + answers[-1][0])

        plt.show()
        new_idx = random.randint(0,1000)
        print("New idx:", new_idx)
        plt.savefig(f'attention_images/image{new_idx}.png')
                
    


    def visualize_txt_attentions(self,outputs, input_tok, output_tok, question):#, input_text, output_text):
        '''tens = 0
        new_input_tok = []
        question = question[-1]
        #print(input_tok[0])
        for idx,x in enumerate(list(encoder_attentions)):

            print("Shapes",x[0].shape, x[1].shape, x[-1].shape)
            #Get the tensors with max attention instead of the first 15 
            x = x[-1][0,-1,:,:len(input_tok[-1])]
            

            if type(tens) == int:
                tens = x
            else:
                tens = tens + x 
        tens = tens/10
        print("Attention shape:", tens.shape)

        max_val, max_idxs = torch.topk(tens,15, dim = 1)
        print("MAx indexes:",max_idxs, max_idxs.shape)
        for idx in max_idxs.squeeze():
            print("individual index",int(idx))
            print("INput tok:", len(input_tok), len(input_tok[0]))
            new_input_tok.append(input_tok[-1][int(idx)])
        print("max_att:", max_val.shape)
        #print("Attention shape:", tens.shape)

        fig = plt.figure(figsize=(15, 5))
        
        attention = max_val.cpu().numpy()
        
        sns.heatmap(attention, cmap='viridis', cbar=False, annot = True, xticklabels = new_input_tok)#, yticklabels = output_tok))#, xticklabels = input_tokens, yticklabels = output_text)
        title = ' '.join(question) + ' '.join(output_tok[-1])
        plt.title(title)

        plt.show()
        new_idx = random.randint(0,1000)
        plt.savefig(f'attention_images/image{new_idx}.png')
        
        total_enc = []
        total_dec = []
        total_cross = []
        
        for enc, dec, cross in zip(encoder_attention, decoder_attention, cross_attention):
            print("encoder:", enc.shape)
            print("decoder:", dec[-1].shape)
            print("cross:", cross[-1].shape)
            enc = enc[-1,:,:,:]
            total_enc.append(enc)
            dec = dec[-1][-1,:,:,:]
            total_dec.append(dec)
            cross = cross[-1][-1,:,:,:]
            total_cross.append(cross)'''

        encoder_attention=outputs.encoder_attentions
        decoder_attention=outputs.decoder_attentions
        cross_attention=outputs.cross_attentions

        print(len(encoder_attention), encoder_attention[0].shape)
        print(len(decoder_attention), decoder_attention[0].shape)
        print(len(cross_attention), cross_attention[0].shape)
        print(encoder_attention[0].shape[-1])
        pad_tok =  ['pad' for _ in range(encoder_attention[0].shape[-1] - len(input_tok[0]))]
        print(len(pad_tok), len(input_tok[0]))
        input_tok = input_tok[0] + pad_tok
        #print("INPUT TOKENS", len(input_tok), input_tok)
        
        print("Output TOKENS", len(output_tok), output_tok)
        #output_tok = [[self.tokenizer(ans)] for ans in output_tok]
        #print(output_tok.keys())

        model_visualization = model_view(
            encoder_attention=outputs.encoder_attentions,
            #decoder_attention=outputs.decoder_attentions[],
            #cross_attention=outputs.cross_attentions,
            encoder_tokens= input_tok,
            #decoder_tokens = output_tok[0], 
            html_action='return'
        )
        new_idx = random.randint(0,1000)
        with open(f"attention_images/head_view{new_idx}.html", 'w') as file:
            file.write(model_visualization.data)
        


    def get_answer_from_model_output(self, input_embeds, attention_mask):
        

        output = self.model.generate(inputs_embeds=input_embeds, attention_mask=attention_mask, output_scores=True, return_dict_in_generate=True, output_attentions=True)
        #print("Shape of decoder embed: ",  len(output['decoder_attentions']),len(output['decoder_attentions'][0]),len(output['decoder_attentions'][0][0]),len(output['cross_attentions'][0][0][0]))
        
        #print("Shape of decoder embed: ", torch.tensor(output['decoder_attentions']).shape)
        
        '''for layer,batch_sample in enumerate(cross_attentions):
            print("Batch.shape:",batch_samp, decoder_attention, cross_attentionle[-1][0,-1,:,:15].shape)
            for head in batch_sample:
                print("Shapes: ",head.shape)
                print(len(output['sequences']))'''
        
        #print(output['decoder_attentions'])
        encoder_attention=output.encoder_attentions
        decoder_attention=output.decoder_attentions
        cross_attention=output.cross_attentions
        
        try:
            pred_answers = self.tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)

        except:
            pred_answers = []
            #print(output['sequences'])
            for out in output['sequences']:
                print(out)
                pred_answers.append(self.tokenizer.decode(out, skip_special_tokens = True))
            print(pred_answers)
        
        pred_answers_conf = model_utils.get_generative_confidence(output)

        return pred_answers, pred_answers_conf, encoder_attention, decoder_attention, cross_attention
