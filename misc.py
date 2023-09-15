import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from sklearn import metrics

class SLMClass(torch.nn.Module):
    def __init__(self, model):
        super(SLMClass, self).__init__()
        self.l1 = model
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
def train_eval(start_epoch, end_epoch, model, train_loader, device, optimizer, loss_fn, logger, save_steps, eval_dataloader, training_state):
    model.train()
    for cur_epoch in range(start_epoch, end_epoch):
        for step,data in enumerate(tqdm(train_loader, total=len(train_loader), desc="Training")):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if step%5000==0:
                print(f'Epoch: {cur_epoch}, Loss:  {loss.item()}')
                logger.info(f"Epoch: {cur_epoch}, Loss:  {loss.item()}")
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % save_steps == 0:
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

                result = validation(eval_dataloader, device, model, logger)
                eval_acc = result['eval_acc']
            
            model.train()

            training_state['dev_acc'].append({'epoch': training_state['epoch'], 'dev_acc': eval_acc})
    return training_state

def validation(eval_dataloader, device, model, logger):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating")):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        
        preds = fin_outputs.argmax(axis=1)
        eval_acc = metrics.accuracy_score(fin_targets, preds)
        f1_score = metrics.f1_score(fin_targets, preds, average="micro")
    result = {
        "eval_acc": round(eval_acc, 4),
        "f1_score": round(f1_score, 4)
    }
    logger.info("***** Eval Ended *****")
    return result
