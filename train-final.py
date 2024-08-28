import pandas as pd
from transformers import (
    T5ForConditionalGeneration,T5TokenizerFast
)
from torch.utils.data import DataLoader,Dataset
import argparse
import torch
from multiprocessing import freeze_support
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from torch.optim import lr_scheduler
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.core import LightningDataModule,LightningModule
from lightning.pytorch.tuner import Tuner

class BorderDataset(Dataset):

    def __init__(self,data):
        self.data=data
        self.data.label=self.data.label.astype(str)
        self.tokenizer=T5TokenizerFast.from_pretrained(r'flant5large')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        source_encoding=self.tokenizer(self.data.text.iloc[index]+'</s>',max_length=560, padding='max_length', truncation=True, return_tensors='pt')
        target_encoding=self.tokenizer(self.data.label.iloc[index]+'</s>',max_length=2, padding='max_length', truncation=True, return_tensors='pt')

        return dict(

        source_ids = source_encoding["input_ids"].squeeze(),
        target_ids = target_encoding["input_ids"].squeeze(),

        source_mask = source_encoding['attention_mask'].squeeze(),
        target_mask = target_encoding['attention_mask'].squeeze(),

        )

class BorderDataModule(LightningDataModule):

    def __init__(
    self,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ): 
        super().__init__()
        self.tokenizer = T5TokenizerFast.from_pretrained(r'flant5large')
        self.train_df = train_df
        self.val_df = val_df

    def setup(self,stage=None):
        self.train_dataset = BorderDataset(self.train_df)
        self.validation_dataset = BorderDataset(self.val_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = 8, shuffle=True, num_workers = 8,persistent_workers=True)

    def val_dataloader(self): 
        return DataLoader(self.validation_dataset, batch_size = 8, num_workers = 8,persistent_workers=True)

class T5FineTuner(LightningModule):
    def __init__(self,learning_rate):
        super(T5FineTuner,self).__init__()
        self.model=T5ForConditionalGeneration.from_pretrained(r'flant5large')
        self.tokenizer=T5TokenizerFast.from_pretrained(r'flant5large')
        self.training_step_outputs=[]
        self.validation_step_outputs=[]
        self.test_step_outputs=[]
        self.validation_step_answer=0
        self.test_step_answer=0
        self.learning_rate=learning_rate

    def forward(self,input_ids,attention_mask=None,decoder_input_ids=None,decoder_attention_mask=None,lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels)
    
    def training_step(self,batch,batch_idx):
        lm_labels=batch['target_ids']
        lm_labels[lm_labels[:,:]==self.tokenizer.pad_token_id]=-100
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=lm_labels
        )
        loss=outputs[0]
        
        self.log('train_loss',loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        self.training_step_outputs.append(outputs[0])
        return loss
    
    def on_train_epoch_end(self):
        num=len(train)
        avg_train_loss = torch.stack([x for x in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
    
    def validation_step(self,batch,batch_idx):
        lm_labels=batch['target_ids']
        lm_labels[lm_labels[:,:]==self.tokenizer.pad_token_id]=-100
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=lm_labels
        )
        loss=outputs[0]
        self.validation_step_answer=self.validation_step_answer+(torch.all(torch.eq(outputs[1].argmax(dim=2),lm_labels),dim=1)).float().sum()
        self.log('val_loss',loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        self.validation_step_outputs.append(outputs[0])
        return loss

    def on_validation_epoch_end(self):
        num=len(val)
        avg_val_loss = torch.stack([x for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()
        acc=self.validation_step_answer/num
        print("********\n")
        print(acc)
        print("********\n")
        tensorboard_logs ={"avg_val_loss":avg_val_loss,"avg_val_acc":acc}
        self.log('val_acc',acc,on_epoch=True,prog_bar=True,logger=True)
        self.validation_step_answer=0
        return {"avg_val_loss": avg_val_loss,"avg_val_acc":acc, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
    
    def test_step(self,batch,batch_idx):
        lm_labels=batch['target_ids']
        lm_labels[lm_labels[:,:]==self.tokenizer.pad_token_id]=-100
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=lm_labels
        )
        loss=outputs[0]
        self.test_step_answer=self.test_step_answer+(torch.all(torch.eq(outputs[1].argmax(dim=2),lm_labels),dim=1)).float().sum()
        self.log('val_loss',loss,on_step=True,on_epoch=True,prog_bar=True,logger=True)
        self.test_step_outputs.append(outputs[0])
        return loss
    
    def on_test_epoch_end(self):
        num=len(val)
        avg_test_loss = torch.stack([x for x in self.test_step_outputs]).mean()
        self.test_step_outputs.clear()
        acc=self.test_step_answer/num
        print("********\n")
        print(acc)
        print("********\n")
        tensorboard_logs ={"avg_test_loss":avg_test_loss,"avg_test_acc":acc}
        self.log('test_acc',acc,on_epoch=True,prog_bar=True,logger=True)
        self.test_step_answer=0
        return {"avg_test_loss": avg_test_loss,"avg_test_acc":acc, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
    
    def predict_step(self,batch):
        input_ids=batch['source_ids']
        attention_mask=batch['source_mask']
        outputs=self.model.generate(input_ids.cuda(),attention_mask=attention_mask.cuda(),max_length=2)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, eps=1e-8,weight_decay=0.01)
        return {
            'optimizer': optimizer,
            'lr_scheduler':{"scheduler":lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=1e-8),"interval":"epoch"}
        }

if __name__ == '__main__':

    freeze_support()

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3,mode='min')
    checkpoint_callback = ModelCheckpoint(
        dirpath='outputcheckpoint', #save at this folder
        filename="flant5large-{epoch:02d}-{val_loss:.2f}", #name for the checkpoint, before i was using "best-checkpoint"
        save_top_k=2, #save all epochs, before it was only the best (1)
        verbose=True, #output something when a model is saved
        monitor="val_loss", #monitor the validation loss
        mode="min" #save the model with minimum validation loss
        )

    logger = TensorBoardLogger(save_dir='tf_dir')
    
    train=pd.read_csv(r'data_train.csv',sep='|')
    val=pd.read_csv(r'data_val.csv',sep='|')
    test_df=pd.read_csv(r'data_test.csv',sep='|')
    data_module=BorderDataModule(train_df=train,val_df=val)
    data_module.setup()
    
    train_dls=DataLoader(BorderDataset(train),batch_size = 8, num_workers = 8,persistent_workers=True)
    val_dls=DataLoader(BorderDataset(val),batch_size = 8, num_workers = 8,persistent_workers=True)
    test_dls=DataLoader(BorderDataset(test_df),batch_size = 8, num_workers = 8,persistent_workers=True)
    


    while 1:
        
        control=int(input('Enter for function: 1-train 2-test  :\n'))
        
        if control==1:
            model=T5FineTuner(learning_rate=1e-4)
            trainer = Trainer(
                max_epochs = 20,
                logger = logger,
                callbacks=[checkpoint_callback,early_stopping_callback]
                )
            trainer.fit(model,datamodule=data_module)
            
        if control==2:
            
            model=T5FineTuner.load_from_checkpoint('outputcheckpoint/flant5large-epoch=01-val_loss=0.21.ckpt',learning_rate=1e-4)
            
            num=len(test_df)
            
            temp=[]
            
            predictions=trainer.predict(model,dataloaders=test_dls)
            a=torch.cat(predictions,dim=0)
            pre=torch.split(a,1,dim=1)[1]
            for data in test_dls:
                temp.append(data['target_ids'])
            b=torch.cat(temp,dim=0)
            label=torch.split(b,1,dim=1)[0]
            res=torch.eq(pre,label)
            print("\n")
            print("********\n")
            print('acc:',res.float().sum()/num)
            print("********\n")

