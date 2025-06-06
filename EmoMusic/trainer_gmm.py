'''
Music FaderNets, GM-VAE model.
'''
import json
import torch
import os
import numpy as np
from MusicVAE import MusicAttrRegGMVAE
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch import nn
from music_dataset import VGMIDIDataset, get_vgmidi
from datetime import datetime
from torch.utils.data import DataLoader
import torch

# some initialization
with open('D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoMusic/MusicVAE.json') as f:
    args = json.load(f)

save_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoMusic/params/'
resume_path = 'D:/PolyU/URIS/Part2_projects/WEMOM_V1/EmoMusic/params/music_attr_vae_reg_gmm_long_v_95.pt'


if not os.path.exists(save_path):
    os.makedirs(save_path)

# ====================== MODELS ===================== #

EVENT_DIMS = 342
RHYTHM_DIMS = 3
NOTE_DIMS = 16
CHROMA_DIMS = 24

model = MusicAttrRegGMVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
                        chroma_dims=CHROMA_DIMS,
                        hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
                        n_step=args['time_step'],
                        n_component=args['num_clusters'])

optimizer = optim.Adam(model.parameters(), lr=args['lr'])


if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')

step, pre_epoch = 0, 0
batch_size = args["batch_size"]
model.train()


# vgmidi dataloaders
print("Loading VGMIDI...")
is_shuffle = True
data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst, label_lst = get_vgmidi()
#----------------------------------------------------------------------------------------
# print("Class distribution in arousal labels:")
# print(Counter(arousal_lst))
#----------------------------------------------------------------------------------------
vgm_train_ds_dist = VGMIDIDataset(data_lst, rhythm_lst, note_density_lst, chroma_lst, arousal_lst, valence_lst, label_lst, mode="train")
vgm_train_dl_dist = DataLoader(vgm_train_ds_dist, batch_size=batch_size, shuffle=is_shuffle, num_workers=0, drop_last=True)
vgm_val_ds_dist = VGMIDIDataset(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, arousal_lst, valence_lst, label_lst, mode="val")
vgm_val_dl_dist = DataLoader(vgm_val_ds_dist, batch_size=batch_size, shuffle=is_shuffle, num_workers=0, drop_last=True)

print("VGMIDI: Train / Test")
print(len(vgm_train_ds_dist), len(vgm_val_ds_dist))
print()


# ====================== TRAINING ===================== #
def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function(out, d,
                r_out, r,
                n_out, n,
                dis,
                qy_x_out,
                logLogit_out,
                step,
                beta=.1,
                y_label=None):
    '''
    Following loss function defined for GMM-VAE:
    Unsupervised: E[log p(x|z)] - sum{l} q(y_l|X) * KL[q(z|x) || p(z|y_l)] - KL[q(y|x) || p(y)]
    Supervised: E[log p(x|z)] - KL[q(z|x) || p(z|y)]
    '''
    # anneal beta
    if step < 1000:
        beta0 = 0
    else:
        beta0 = min((step - 1000) / 1000 * beta, beta) 

    # Reconstruction loss
    CE_X = F.nll_loss(out.view(-1, out.size(-1)),
                    d.view(-1), reduction='mean')
    CE_R = F.nll_loss(r_out.view(-1, r_out.size(-1)),
                    r.view(-1), reduction='mean')
    CE_N = F.nll_loss(n_out.view(-1, n_out.size(-1)),
                    n.view(-1), reduction='mean')

    CE = 5 * CE_X + CE_R + CE_N

    # package output
    dis_r, dis_n = dis
    qy_x_r, qy_x_n = qy_x_out
    logLogit_qy_x_r, logLogit_qy_x_n = logLogit_out
    
    # Debug: Print qy_x_r and qy_x_n values
    #print(f"Step {step} - qy_x_r: {qy_x_r}, qy_x_n: {qy_x_n}")
    # KLD latent and class loss
    kld_lat_r_total, kld_lat_n_total = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
    # Debug: Check the range of y_labels:
    #print(f"Step {step} - y_label: {y_label}")
    #print(f"y_label min: {y_label.min()}, y_label max: {y_label.max()}")
    #print(f"y_label dtype: {y_label.dtype}")

    mu_pz_y_r, var_pz_y_r = model.mu_r_lookup(y_label.cuda().long()), model.logvar_r_lookup(y_label.cuda().long()).exp_()
    dis_pz_y_r = Normal(mu_pz_y_r, var_pz_y_r)
    kld_lat_r = torch.mean(kl_divergence(dis_r, dis_pz_y_r), dim=-1)

    mu_pz_y_n, var_pz_y_n = model.mu_n_lookup(y_label.cuda().long()), model.logvar_n_lookup(y_label.cuda().long()).exp_()
    dis_pz_y_n = Normal(mu_pz_y_n, var_pz_y_n)
    kld_lat_n = torch.mean(kl_divergence(dis_n, dis_pz_y_n), dim=-1)

    # Debug: Check model lookup
    #print(f"mu_r_lookup size: {model.mu_r_lookup.weight.size()}")
    #print(f"logvar_r_lookup size: {model.logvar_r_lookup.weight.size()}")

    kld_lat_r_total, kld_lat_n_total = kld_lat_r.mean(), kld_lat_n.mean()

    label_clf_loss = nn.CrossEntropyLoss()(qy_x_r, y_label.cuda().long()) + \
                        nn.CrossEntropyLoss()(qy_x_n, y_label.cuda().long())
    loss = CE + beta0 * (kld_lat_r_total + kld_lat_n_total) + label_clf_loss
    #------------------------------------------------------------------------------------------- 
    #print(f"Step {step} - CE_X: {CE_X.item()}, CE_R: {CE_R.item()}, CE_N: {CE_N.item()}")
    #print(f"Step {step} - KLD_lat_r_total: {kld_lat_r_total.item()}, KLD_lat_n_total: {kld_lat_n_total.item()}")
    #print(f"Step {step} - Label Clf Loss: {label_clf_loss.item()}")
    # ---------------------------------------------------------------------------------
    return loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total


def latent_regularized_loss_function(z_out, r, n):
    # regularization loss - Pati et al. 2019
    z_r, z_n = z_out

    z_r_new = z_r
    z_n_new = z_n

    # rhythm regularized
    r_density = r
    D_attr_r = torch.from_numpy(np.subtract.outer(r_density, r_density)).cuda().float()
    D_z_r = z_r_new[:, 0].reshape(-1, 1) - z_r_new[:, 0]
    l_r = torch.nn.MSELoss(reduction="mean")(torch.tanh(D_z_r), torch.sign(D_attr_r))
        
    n_density = n
    D_attr_n = torch.from_numpy(np.subtract.outer(n_density, n_density)).cuda().float()
    D_z_n = z_n_new[:, 0].reshape(-1, 1) - z_n_new[:, 0]
    l_n = torch.nn.MSELoss(reduction="mean")(torch.tanh(D_z_n), torch.sign(D_attr_n))

    return l_r, l_n


def train(step, d_oh, r_oh, n_oh, d, r, n, c, r_density, n_density, y_label=None):
    
    optimizer.zero_grad()
    # pdb.set_trace()
    res = model(d_oh, r_oh, n_oh, c)

    # package output
    output, dis, z_out, logLogit_out, qy_x_out, y_out = res
    out, r_out, n_out, _, _ = output

    # calculate gmm loss
    loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total = loss_function(out, d,
                                        r_out, r,
                                        n_out, n,
                                        dis,
                                        qy_x_out,
                                        logLogit_out,
                                        step,
                                        beta=args['beta'],
                                        y_label=y_label)
    
    # calculate latent regularization loss
    l_r, l_n = torch.Tensor([0]), torch.Tensor([0])
    l_r, l_n = latent_regularized_loss_function(z_out, r_density, n_density)
    loss += l_r + l_n
    
  
    loss.backward()

    

    #-----------------------------------------------------------------------
    # Degug
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Grad for {name}: {param.grad.norm()}")
    #-----------------------------------------------------------------------

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    step += 1

    kld_latent = kld_lat_r_total + kld_lat_n_total
    output = loss.item(), CE_X.item(), CE_R.item(), CE_N.item(), l_r.item(), l_n.item(), kld_latent.item()
    return step, output


def evaluate(step, d_oh, r_oh, n_oh, d, r, n, c, r_density, n_density, y_label=None):

    res = model(d_oh, r_oh, n_oh, c)

    # package output
    output, dis, z_out, logLogit_out, qy_x_out, y_out = res
    out, r_out, n_out, _, _ = output
    z_r, z_n = z_out

    # calculate gmm loss
    loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total = loss_function(out, d,
                                        r_out, r,
                                        n_out, n,
                                        dis,
                                        qy_x_out,
                                        logLogit_out,
                                        step,
                                        beta=args['beta'],
                                        y_label=y_label)
    
    # calculate latent regularization loss
    l_r, l_n = torch.Tensor([0]), torch.Tensor([0])
    l_r, l_n = latent_regularized_loss_function(z_out, r_density, n_density)
    loss += l_r + l_n

    kld_latent = kld_lat_r_total + kld_lat_n_total
    
    output = loss.item(), CE_X.item(), CE_R.item(), CE_N.item(), l_r.item(), l_n.item(), kld_latent.item()
    return output


def convert_to_one_hot(input, dims):
    if len(input.shape) > 1:
        input_oh = torch.zeros((input.shape[0], input.shape[1], dims)).cuda()
        input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
    else:
        input_oh = torch.zeros((input.shape[0], dims)).cuda()
        input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
    return input_oh


def training_phase(step):


    print("D - Data, R - Rhythm, N - Note, RD - Reg. Rhythm, ND- Reg. Note, KLD-L: KLD Latent, KLD-C: KLD Class")
    for i in range(start_epoch, args['n_epochs'] + 1):
        print("Epoch {} / {}".format(i, args['n_epochs']))

        # =================== TRAIN VGMIDI ======================== #

        batch_loss, batch_test_loss = 0, 0
        b_CE_X, b_CE_R, b_CE_N = 0, 0, 0
        t_CE_X, t_CE_R, t_CE_N = 0, 0, 0
        b_l_r, b_l_n, t_l_r, t_l_n = 0, 0, 0, 0
        b_kld_latent, b_kld_class, t_kld_latent, t_kld_class  = 0, 0, 0, 0
        
        # train on vgmidi
        for j, x in enumerate(vgm_train_dl_dist):

            d, r, n, c, a, v, l, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)
            r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
            n_oh = convert_to_one_hot(n, NOTE_DIMS)
            # c_oh = convert_to_one_hot(c, CHROMA_DIMS)

            step, loss = train(step, d_oh, r_oh, n_oh, d, r, n, c, r_density, n_density, y_label=l)
            loss, CE_X, CE_R, CE_N, l_r, l_n, kld_latent = loss
            batch_loss += loss

            b_CE_X += CE_X
            b_CE_R += CE_R
            b_CE_N += CE_N
            b_l_r += l_r
            b_l_n += l_n
            b_kld_latent += kld_latent

            print('batch loss {}/{}: {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(j, len(vgm_train_dl_dist), loss, CE_X, CE_R, CE_N, l_r, l_n, kld_latent))
        
        # evaluate on vgmidi
        for j, x in enumerate(vgm_val_dl_dist):
            d, r, n, c, a, v, l, r_density, n_density = x
            d, r, n, c, l = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float(), l.cuda().long()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)
            r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
            n_oh = convert_to_one_hot(n, NOTE_DIMS)
            # c_oh = convert_to_one_hot(c, CHROMA_DIMS)

            loss = evaluate(step - 1, d_oh, r_oh, n_oh, d, r, n, c, r_density, n_density, y_label=l)
            loss, CE_X, CE_R, CE_N, l_r, l_n, kld_latent= loss
            batch_test_loss += loss
            
            t_CE_X += CE_X
            t_CE_R += CE_R
            t_CE_N += CE_N
            t_l_r += l_r
            t_l_n += l_n
            t_kld_latent += kld_latent
        
        print('epoch loss: {:.5f}  {:.5f}'.format(batch_loss / len(vgm_train_dl_dist),
                                                  batch_test_loss / len(vgm_val_dl_dist)))

        print("train loss by term - D: {:.4f} R: {:.4f} N: {:.4f} RD: {:.4f} ND: {:.4f} KLD-L: {:.4f} ".format(
            b_CE_X / len(vgm_train_dl_dist), b_CE_R / len(vgm_train_dl_dist), 
            b_CE_N / len(vgm_train_dl_dist),
            b_l_r / len(vgm_train_dl_dist), b_l_n / len(vgm_train_dl_dist),
            b_kld_latent / len(vgm_train_dl_dist)
        ))
        print("test loss by term - D: {:.4f} R: {:.4f} N: {:.4f} RD: {:.4f} ND: {:.4f} KLD-L: {:.4f} ".format(
            t_CE_X / len(vgm_val_dl_dist), t_CE_R / len(vgm_val_dl_dist), 
            t_CE_N / len(vgm_val_dl_dist),
            t_l_r / len(vgm_val_dl_dist), t_l_n / len(vgm_val_dl_dist),
            t_kld_latent / len(vgm_val_dl_dist)
        ))

        if i % 5 == 0:
            save_epoch_path = save_path + ("{}.pt".format(args['name'] + "_" + str(i)))
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': batch_loss
            }, save_epoch_path)
            print("Saving model to ... ", save_epoch_path)


    timestamp = str(datetime.now())

    save_path_timing = save_path + ("{}.pt".format(args['name'] + "_" + timestamp))
    torch.save({
        'epoch': 100,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss
    }, save_path_timing)
    print('Model saved as {}!'.format(save_path_timing))


# def evaluation_phase():
#     print("Evaluate")
#     if torch.cuda.is_available():
#         model.cuda()

#     if os.path.exists(save_path):
#         print("Loading {}".format(save_path))
#         model.load_state_dict(torch.load(save_path))
    
#     def run(dl, is_vgmidi=False):
        
#         t_CE_X, t_CE_R, t_CE_N = 0, 0, 0
#         t_l_r, t_l_n = 0, 0
#         t_kld_latent, t_kld_class = 0, 0
#         t_acc_x, t_acc_r, t_acc_n, t_acc_a_r, t_acc_a_n = 0, 0, 0, 0, 0
#         data_len = 0

#         for i, x in tqdm(enumerate(dl), total=len(dl)):
#             d, r, n, c, a, v, r_density, n_density = x
#             d, r, n, c = d.cuda().long(), r.cuda().long(), \
#                          n.cuda().long(), c.cuda().long()

#             d_oh = convert_to_one_hot(d, EVENT_DIMS)
#             r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
#             n_oh = convert_to_one_hot(n, NOTE_DIMS)

#             res = model(d_oh, r_oh, n_oh, c)

#             # package output
#             output, dis, z_out, logLogit_out, qy_x_out, y_out = res
#             out, r_out, n_out, _, _ = output
#             z_r, z_n = z_out

#             if not is_vgmidi:
#                 # calculate gmm loss
#                 loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total, \
#                     kld_cls_r, kld_cls_n = loss_function(out, d,
#                                                     r_out, r,
#                                                     n_out, n,
#                                                     dis,
#                                                     qy_x_out,
#                                                     logLogit_out,
#                                                     step,
#                                                     beta=args['beta'])
            
#             else:
#                 # calculate gmm loss
#                 loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total, \
#                     kld_cls_r, kld_cls_n = loss_function(out, d,
#                                                     r_out, r,
#                                                     n_out, n,
#                                                     dis,
#                                                     qy_x_out,
#                                                     logLogit_out,
#                                                     step,
#                                                     beta=args['beta'],
#                                                     is_supervised=True,
#                                                     y_label=a)
            
#             # calculate latent regularization loss
#             l_r, l_n = latent_regularized_loss_function(z_out, r_density, n_density)

#             # adversarial loss
#             kld_latent, kld_class = kld_lat_r_total.item() +  kld_lat_n_total.item(), \
#                                     kld_cls_r.item() + kld_cls_n.item()
            
#             t_CE_X += CE_X
#             t_CE_R += CE_R
#             t_CE_N += CE_N
#             t_l_r += l_r.item()
#             t_l_n += l_n.item()
#             t_kld_latent += kld_latent
#             t_kld_class += kld_class
            
#             # calculate accuracy
#             def acc(a, b, t, trim=False):
#                 a = torch.argmax(a, dim=-1).squeeze().cpu().detach().numpy()
#                 b = b.squeeze().cpu().detach().numpy()

#                 b_acc = 0
#                 for i in range(len(a)):
#                     a_batch = a[i]
#                     b_batch = b[i]

#                     if trim:
#                         b_batch = np.trim_zeros(b_batch)
#                         a_batch = a_batch[:len(b_batch)]

#                     correct = 0
#                     for j in range(len(a_batch)):
#                         if a_batch[j] == b_batch[j]:
#                             correct += 1
#                     acc = correct / len(a_batch)
#                     b_acc += acc
                
#                 return b_acc

#             acc_x, acc_r, acc_n = acc(out, d, "d", trim=True), \
#                                   acc(r_out, r, "r"), acc(n_out, n, "n")
#             data_len += out.shape[0]

#             if is_vgmidi:
#                 qy_x_r, qy_x_n = qy_x_out
#                 qy_x_r, qy_x_n = torch.argmax(qy_x_r, axis=-1).cpu().detach().numpy(), \
#                                 torch.argmax(qy_x_n, axis=-1).cpu().detach().numpy()
#                 acc_q_x_r = accuracy_score(a.cpu().detach().numpy(), qy_x_r)
#                 acc_q_x_n = accuracy_score(a.cpu().detach().numpy(), qy_x_n)
#             else:
#                 acc_q_x_r, acc_q_x_n = 0, 0

#             t_acc_x += acc_x
#             t_acc_r += acc_r
#             t_acc_n += acc_n
#             t_acc_a_r += acc_q_x_r
#             t_acc_a_n += acc_q_x_n

#         # Print results
#         print("CE: {:.4}  {:.4}  {:.4}".format(t_CE_X / len(dl),
#                                                     t_CE_R / len(dl), 
#                                                     t_CE_N / len(dl)))
        
#         print("Regularized: {:.4}  {:.4}".format(t_l_r / len(dl),
#                                                 t_l_n / len(dl)))

#         # print("Adversarial: {:.4}  {:.4}".format(t_l_adv_r / len(dl),
#         #                                         t_l_adv_n / len(dl)))
        
#         print("Acc: {:.4}  {:.4}  {:.4}  {:.4}  {:.4}".format(t_acc_x / data_len,
#                                                             t_acc_r / data_len, 
#                                                             t_acc_n / data_len,
#                                                             t_acc_a_r / data_len,
#                                                             t_acc_a_n / data_len))

#     # dl = DataLoader(train_ds_dist, batch_size=128, shuffle=False, num_workers=0)
#     # run(dl)
#     # dl = DataLoader(test_ds_dist, batch_size=128, shuffle=False, num_workers=0)
#     # run(dl)
#     # dl = DataLoader(vgm_train_ds_dist, batch_size=32, shuffle=False, num_workers=0)
#     # run(dl, is_vgmidi=True)
#     # dl = DataLoader(vgm_test_ds_dist, batch_size=32, shuffle=False, num_workers=0)
#     # run(dl, is_vgmidi=True)

if os.path.exists(resume_path):
    print(f"Resuming training from {resume_path}...")
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    step = checkpoint.get('step', 0)
    print(f"Resumed from epoch {start_epoch}")
else:
    print("No checkpoint found. Starting training from scratch.")
    start_epoch = 1

training_phase(step)
# evaluation_phase()

