
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from TextVAE import TextVAE
from text_dataset import TextDataset, get_text_data
import json
import os
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from utils import ids_to_words
import numpy as np


# Load configuration
with open('D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoText/TextVAE_config.json') as f:
    args = json.load(f)

id2word = np.load('D:/PolyU/URIS/Part2_projects/WEMOM_V1/Data/All_Data/id2word_text.npy', allow_pickle=True).item()

# Define hyperparameters
batch_size = args['batch_size']
embedding_size = args['embedding_size']
vocab_size = args['vocab_size']
hidden_size = args['hidden_size']
num_layers = args['num_layers']
dropout = args['dropout']
clip_grad_norm = args['clip_grad_norm']

save_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoText/params/'
resume_path = save_path + args['name'] + '_latest.pt'

if not os.path.exists(save_path):
    os.makedirs(save_path)


# Load data
sentence_lst, label_lst = get_text_data()
train_dataset = TextDataset(sentence_lst, label_lst, mode="train")
test_dataset = TextDataset(sentence_lst, label_lst, mode="val")

label_counts = Counter(label_lst)
total_count = sum(label_counts.values())
class_weights = [total_count / label_counts[i] for i in range(len(label_counts))]

sample_weights = [1.0 / label_counts[label] for label in train_dataset.label.cpu().numpy()]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

train_label_counts = Counter(train_dataset.label.cpu().numpy())
val_label_counts = Counter(test_dataset.label.cpu().numpy())

print("Training set label distribution:", train_label_counts)
print("Validation set label distribution:", val_label_counts)



# Initialize the model
model = TextVAE(
    vocab_size=vocab_size,
    embed_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout
)

if torch.cuda.is_available():
    print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


# Define loss function
def loss_function(logit, text_target, mean, std, logits, labels, KL_weight, class_weights=class_weights):
    criterion = nn.CrossEntropyLoss(ignore_index=args["PAD_INDEX"])
    reconstruction_loss = criterion(logit, text_target)
    kl_loss = 0.5 * (-torch.log(std ** 2) + mean ** 2 + std ** 2 - 1).mean()
    classification_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).cuda())
    classification_loss = classification_loss_fn(logits, labels)
    loss = reconstruction_loss + kl_loss * KL_weight + classification_loss * args['alpha']
    return loss, reconstruction_loss, kl_loss, classification_loss

# Training phase
def training_phase(step):
    print("Starting training...")
    for epoch in range(step, args['num_epochs'] + 1):
        print(f"Epoch {epoch} started. Current learning rate:")
        for param_group in optimizer.param_groups:
            print(f"  {param_group['lr']}")

        model.train()
        epoch_loss = 0
        for i, (sentence, labels) in enumerate(train_data):
            
            sentence, labels = sentence.cuda(), labels.cuda()
            labels = labels.long()
            optimizer.zero_grad()

            logit, mean, std, logits = model(sentence)
            target_output = sentence[:, 1:]  # Remove <sos>
            # print(logit, mean, std, logits)

            logit = logit.reshape(-1, logit.size(-1))  
            target_output = target_output.reshape(-1)

            # KL Annealing
            kl_weight = min(1.0, epoch / 1.0) * args["lambda"]  # 在前 10 个 epoch 内逐步增加 KL 权重
            loss, reconstruction_loss, kl_loss, classification_loss = loss_function(
            logit, target_output, mean, std, logits, labels, kl_weight, class_weights
            )

            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad.norm().item()}")
            # nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            # optimizer.step()

            step += 1
            epoch_loss += loss.item()

            if i % 10 == 0:
                # print(f"Logits distribution (softmax): {logits.softmax(dim=-1).mean(dim=0).tolist()}")
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}, "
                      f"Reconstruction: {reconstruction_loss.item():.4f}, "
                      f"KL: {kl_loss.item():.4f}, Classification: {classification_loss.item():.4f}")
                # print(f"Predicted labels: {logits.argmax(dim=1)[:10].tolist()}")
                # print(f"True labels: {labels[:10].tolist()}")
        print(f"Epoch {epoch} completed. Average Loss: {epoch_loss / len(train_data):.4f}")

        # Save checkpoint
        if epoch % 5 == 0:
            print(f"Saving checkpoint at epoch {epoch}...")
            save_checkpoint(epoch, step, model, optimizer, epoch_loss)

        # Evaluate on validation set
        eval_loss = evaluation_phase()
        scheduler.step(eval_loss)
        print(f"Validation Loss: {eval_loss:.4f}")

# Evaluation phase
def evaluation_phase():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for sentence, labels in test_data:
            sentence, labels = sentence.cuda(), labels.cuda()
            labels = labels.long()

            logit, mean, std, logits = model(sentence)

            logit = logit.reshape(-1, logit.size(-1)) 
            target_output = sentence[:, 1:]  
            target_output = target_output.reshape(-1)

            loss, _, _, _ = loss_function(
                logit, target_output.reshape(-1), mean, std, logits, labels, KL_weight=1.0,class_weights=class_weights)
        
            total_loss += loss.item()
            decoder_output = logit.view(sentence.size(0), -1, logit.size(-1))
            predicted_ids = decoder_output.argmax(dim=-1) 

            for i in range(min(2, sentence.size(0))):  # 只打印前2个样本
                print("Original ids:", sentence[i, 1:].tolist())
                print("Decoded ids :", predicted_ids[i].tolist())
                print("Original:", " ".join(ids_to_words(sentence[i, 1:].tolist(), id2word)))
                print("Decoded :", " ".join(ids_to_words(predicted_ids[i].tolist(), id2word)))

    return total_loss / len(test_data)

# Save checkpoint
def save_checkpoint(epoch, step, model, optimizer, loss):
    save_path_epoch = save_path + f"{args['name']}_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path_epoch)
    print(f"Checkpoint saved: {save_path_epoch}")

    # Save latest checkpoint
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, resume_path)
    print(f"Latest checkpoint saved: {resume_path}")

# Resume training if checkpoint exists
# if os.path.exists(resume_path):
#     print(f"Resuming training from {resume_path}...")
#     checkpoint = torch.load(resume_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1
#     step = checkpoint.get('step', 0)
#     print(f"Resumed from epoch {start_epoch}")
# else:
#     print("No checkpoint found. Starting training from scratch.")
#     start_epoch = 1
#     step = 0

# Start training
step = 0
training_phase(step)
evaluation_phase()