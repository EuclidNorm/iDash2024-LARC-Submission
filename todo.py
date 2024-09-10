import numpy as np
import torch
import math
from torch.utils.data import random_split
from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    BaselineLoss,

    get_nb_max_rounds,
    FedTcgaBrca
)

def softmax_func(weights):
    exp_terms = []
    for i in range(len(weights)):
        exp_terms.append(math.exp(weights[i]))
    output = np.zeros((len(weights)))
    summer = sum(exp_terms)
    for i in range(len(output)):
        output[i] = exp_terms[i] / summer
    return output

def cal_entropy(combined_weights, index_i):
    num_clients=6
    torch.no_grad()
    return_dataset = []
    val_dataset = []
    dataset_local = FedTcgaBrca(center=index_i)

    mtx = np.reshape(dataset_local[0][0].numpy(), (1, 39))
    for i in range(1, len(dataset_local)):
        mtx = np.concatenate((mtx, np.reshape(dataset_local[i][0].numpy(), (1, 39))), axis=0)
    gt = np.reshape(dataset_local[0][1].numpy(), (1, 2))
    for i in range(1, len(dataset_local)):
        gt = np.concatenate((gt, np.reshape(dataset_local[i][1].numpy(), (1, 2))))
    w = np.array(combined_weights[0][0])
    b = combined_weights[1][0]
    scores = []
    for i in range(len(dataset_local)):
        scores.append(np.dot(w, mtx[i]))
    scores = np.array(scores)
    scores = np.reshape(scores, (len(scores), 1))
    scores = torch.from_numpy(scores)

    loss_func = BaselineLoss()
    gt = torch.from_numpy(gt)
    loss = loss_func(scores, gt)

    return loss

def adam_update(delta_weights,current_m,current_v,beta1,beta2,adam_LR,tauarray):
    next_mw = np.zeros((1, len(current_m[0])))
    next_vw = np.zeros((1, len(current_v[0])))
    next_mb = np.zeros((1))
    next_vb = np.zeros((1))
    next_m = []
    next_m.append(next_mw)
    next_m.append(next_mb)
    next_v = []
    next_v.append(next_vw)
    next_v.append(next_vb)
    updates = []
    updates.append(np.zeros((1, 39)))
    updates.append(np.zeros((1)))
    for param_idx in range(len(current_m)):
        next_m[param_idx] = (
                beta1 * current_m[param_idx]
                + (1 - beta1) * delta_weights[param_idx]
        )

    for param_idx in range(len(current_v)):
        next_v[param_idx] = (
                beta2 * current_v[param_idx]
                + (1 - beta2)
                * delta_weights[param_idx]
                * delta_weights[param_idx]
        )

    for param_idx in range(len(updates)):
        updates[param_idx] = (
                adam_LR
                * next_m[param_idx]
                / (np.sqrt(next_v[param_idx]) + tauarray[param_idx])
        )
    return updates

def solution(local_updates,previous_weights,initial_state,current_m,current_v,beta1,beta2,adam_LR,tauarray):   #The last 6 inputs are related to Adam optimizer
    q=19
    b=0.5


    aggregated_delta_weights = [
        None for _ in range(len(local_updates[0]["updates"]))
    ]
    for idx_weight in range(len(local_updates[0]["updates"])):
        aggregated_delta_weights[idx_weight] = sum(
            [
                local_updates[idx_client]["updates"][idx_weight]
                *previous_weights[idx_client]
                for idx_client in range(len(local_updates))
            ]
        )
    delta_losses=np.zeros((6))
    #Here the initial aggregated delta weights are broadcasted to 6 clients together with respective previous weights. Here we calculate the loss.
    for idx_client in range(6):

        delta_weights_client_i_nonlocal = []
        delta_weights_client_i_nonlocal.append(
            aggregated_delta_weights[0] - local_updates[idx_client]["updates"][0] * previous_weights[idx_client])
        delta_weights_client_i_nonlocal.append(
            aggregated_delta_weights[1] - local_updates[idx_client]["updates"][1] * previous_weights[idx_client])
        update_nonlocal=adam_update(delta_weights_client_i_nonlocal,current_m,current_v,beta1,beta2,adam_LR,tauarray)
        weights_nonlocal_w=initial_state[0]+update_nonlocal[0]
        weights_nonlocal_b=initial_state[1]+update_nonlocal[1]
        weights_nonlocal=[]
        weights_nonlocal.append(weights_nonlocal_w)
        weights_nonlocal.append(weights_nonlocal_b)

        delta_weights_client_i_local = []
        delta_weights_client_i_local.append(local_updates[idx_client]["updates"][0] * previous_weights[idx_client])
        delta_weights_client_i_local.append(local_updates[idx_client]["updates"][1] * previous_weights[idx_client])
        update_local=adam_update(delta_weights_client_i_local,current_m,current_v,beta1,beta2,adam_LR,tauarray)
        weights_local_w=initial_state[0]+update_local[0]
        weights_local_b=initial_state[1]+update_local[1]
        weights_local=[]
        weights_local.append(weights_local_w)
        weights_local.append(weights_local_b)

        loss_nonlocal=cal_entropy(weights_nonlocal,idx_client)
        loss_local=cal_entropy(weights_local,idx_client)

        delta_loss=loss_nonlocal-loss_local
        delta_losses[idx_client]=delta_loss*q

    #Here we compute the weights
    softmax_scores=softmax_func(delta_losses)
    softmax_scores/=max(softmax_scores)
    current_weights=(softmax_scores+b)/(1+b)


    #Aggregation is finally done using weights computed above
    aggregated_delta_weights1 = [
        None for _ in range(len(local_updates[0]["updates"]))
    ]
    for idx_weight in range(len(local_updates[0]["updates"])):
        aggregated_delta_weights1[idx_weight] = sum(
            [
                local_updates[idx_client]["updates"][idx_weight]
                * current_weights[idx_client]
                for idx_client in range(len(local_updates))
            ]
        )
    return aggregated_delta_weights1,current_weights
