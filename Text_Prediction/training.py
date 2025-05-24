'''training settings'''

from transformers import TrainingArguments, DataCollatorWithPadding, Trainer
from metrics import compute_metrics
import torch
import pandas as pd
    
class CustomTrainerMSE(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        labels = inputs.get("labels")
        inputs.pop('labels', None) 

        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = torch.nn.MSELoss()      
        mse_loss = loss_fct(logits.view(-1), labels.view(-1)) 
        loss = mse_loss
        # print(loss)
        return (loss, outputs) if return_outputs else loss
    
def training(model, params, dataset, preds_dir):
    output_dir = '/content/drive/MyDrive/text_VA_prediction/'
    model_dir = '/content/drive/MyDrive/text_VA_prediction/'
    
    batch_size = params['batch_size']
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=batch_size, 
        num_train_epochs=params['train_epochs'],
        learning_rate=params['lr'], 
        weight_decay=params['weight_decay'],
        group_by_length=True,
        evaluation_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=params['warmup_ratio'],
        ) 
        
    
    print("Starting")

    train_data = dataset[0]
    val_data = dataset[1]
    
    data_collator = DataCollatorWithPadding(train_data.tokenizer)
   
    trainer = CustomTrainerMSE(
    model,
    training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data,    
    tokenizer=train_data.tokenizer,
    compute_metrics=compute_metrics,
    optimizers = torch.optim.AdamW
    optimizers=(optimizer, self.lr_scheduler)
    )
           
    trainer.train()
    
    preds = trainer.predict(val_data)
    preds_df = pd.DataFrame(preds.predictions)
    run_metrics = preds.metrics
    preds_df.to_csv(preds_dir +  "/predictions_fold.csv")      
    with open(preds_dir + '/fold_metrics.csv', 'w') as fa:     
        for key in run_metrics.keys():
            fa.write("%s,%s\n"%(key,run_metrics[key]))
    fa.close()

    # trainer.save_model(model_dir)  
    

