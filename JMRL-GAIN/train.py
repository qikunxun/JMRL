import time
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from transformers.optimization import AdamW
from config import *
from data import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from models.GAIN import GAIN_GloVe, GAIN_BERT
from test import test
from utils import Accuracy, get_cuda, logging, print_params

matplotlib.use('Agg')

def cal_loss(labels, predictions, relation_mask):
    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
    th_label[0, :, 0] = 1.0
    labels[0, :, 0] = 0.0

    p_mask = labels + th_label
    n_mask = 1 - labels
    # Rank positive classes to TH
    logit1 = predictions - (1 - p_mask) * 1e30
    loss1 = -(torch.log_softmax(logit1, dim=-1) * labels).sum(-1) * relation_mask

    # Rank TH to negative classes
    logit2 = predictions - (1 - n_mask) * 1e30
    loss2 = -(torch.log_softmax(logit2, dim=-1) * th_label).sum(-1) * relation_mask

    # Sum two parts
    loss = loss1 + loss2
    loss = loss.mean()
    return loss

def get_label(logits, num_labels=-1):
    th_logit = logits[:, :, 0].unsqueeze(-1)
    output = torch.zeros_like(logits).to(logits)
    mask = (logits > th_logit)
    if num_labels > 0:
        top_v, _ = torch.topk(logits, num_labels, dim=-1)
        top_v = top_v[:, :, -1]
        mask = (logits >= top_v.unsqueeze(-1)) & mask
    output[mask] = 1.0
    output[:, :, 0] = (output.sum(-1) == 0.).to(logits)
    return output

def train(opt):
    if opt.use_model == 'bert':
        # datasets
        train_set = BERTDGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                     opt=opt)
        dev_set = BERTDGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='dev',
                                   instance_in_train=train_set.instance_in_train, opt=opt)

        # dataloaders
        train_loader = DGLREDataloader(train_set, batch_size=opt.batch_size, shuffle=True,
                                       negativa_alpha=opt.negativa_alpha)
        dev_loader = DGLREDataloader(dev_set, batch_size=opt.test_batch_size, dataset_type='dev')

        model = GAIN_BERT(opt)

    elif opt.use_model == 'bilstm':
        # datasets
        train_set = DGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                 opt=opt)
        dev_set = DGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='dev',
                               instance_in_train=train_set.instance_in_train, opt=opt)

        # dataloaders
        train_loader = DGLREDataloader(train_set, batch_size=opt.batch_size, shuffle=True,
                                       negativa_alpha=opt.negativa_alpha)
        dev_loader = DGLREDataloader(dev_set, batch_size=opt.test_batch_size, dataset_type='dev')

        model = GAIN_GloVe(opt)
    else:
        assert 1 == 2, 'please choose a model from [bert, bilstm].'

    print(model.parameters)
    print_params(model)

    start_epoch = 1
    pretrain_model = opt.pretrain_model
    lr = opt.lr
    model_name = opt.model_name

    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load model from {}'.format(pretrain_model))
        start_epoch = chkpt['epoch'] + 1
        lr = chkpt['lr']
        logging('resume from epoch {} with lr {}'.format(start_epoch, lr))
    else:
        logging('training from scratch with lr {}'.format(lr))

    model = get_cuda(model)

    if opt.use_model == 'bert':
        diff_layer = ['diff']
        bert_layer = ['bert']

        optimizer = optim.AdamW([
            {'params': model.bert.parameters(), },
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in diff_layer + bert_layer)],
             "lr": 1e-3, 'weight_decay': opt.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in diff_layer)], "lr": 3e-1},
        ], lr=lr)

    else:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                eps=1e-6)

    BCE = nn.BCEWithLogitsLoss(reduction='none')

    if opt.coslr:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(opt.epoch // 4) + 1)

    checkpoint_dir = opt.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    fig_result_dir = opt.fig_result_dir
    if not os.path.exists(fig_result_dir):
        os.mkdir(fig_result_dir)

    best_ign_auc = 0.0
    best_ign_f1 = 0.0
    best_epoch = 0

    model.train()

    global_step = 0
    total_loss = 0

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.title('Precision-Recall')
    plt.grid(True)

    acc_NA, acc_not_NA, acc_total = Accuracy(), Accuracy(), Accuracy()
    logging('begin..')

    for epoch in range(start_epoch, opt.epoch + 1):
        start_time = time.time()
        for acc in [acc_NA, acc_not_NA, acc_total]:
            acc.clear()

        for ii, d in enumerate(train_loader):
            relation_multi_label = d['relation_multi_label']
            relation_mask = d['relation_mask']
            relation_label = d['relation_label']

            predictions, predictions_rule = model(words=d['context_idxs'],
                                src_lengths=d['context_word_length'],
                                mask=d['context_word_mask'],
                                entity_type=d['context_ner'],
                                entity_id=d['context_pos'],
                                mention_id=d['context_mention'],
                                distance=None,
                                entity2mention_table=d['entity2mention_table'],
                                graphs=d['graphs'],
                                h_t_pairs=d['h_t_pairs'],
                                relation_mask=relation_mask,
                                path_table=d['path_table'],
                                entity_graphs=d['entity_graphs'],
                                ht_pair_distance=d['ht_pair_distance']
                                )
            loss_1 = torch.sum(BCE(predictions, relation_multi_label) * relation_mask.unsqueeze(2)) / (
                    opt.relation_nums * torch.sum(relation_mask))
            loss_2 = torch.sum(BCE(predictions_rule, relation_multi_label) * relation_mask.unsqueeze(2)) / (
                    opt.relation_nums * torch.sum(relation_mask))

            loss = opt.lambda_al * loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            print(loss, loss_1, loss_2)
            if opt.clip != -1:
                nn.utils.clip_grad_value_(model.parameters(), opt.clip)
            optimizer.step()
            if opt.coslr:
                scheduler.step(epoch)

            output = torch.argmax(predictions_rule, dim=-1)
            output = output.data.cpu().numpy()
            relation_label = relation_label.data.cpu().numpy()

            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    label = relation_label[i][j]
                    if label < 0:
                        break

                    is_correct = (output[i][j] == label)
                    if label == 0:
                        acc_NA.add(is_correct)
                    else:
                        acc_not_NA.add(is_correct)

                    acc_total.add(is_correct)

            global_step += 1
            total_loss += loss.item()

            log_step = opt.log_step
            if global_step % log_step == 0:
                cur_loss = total_loss / log_step
                elapsed = time.time() - start_time
                logging(
                    '| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:5.3f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(
                        epoch, global_step, elapsed * 1000 / log_step, cur_loss * 1000, acc_NA.get(), acc_not_NA.get(),
                        acc_total.get()))
                total_loss = 0
                start_time = time.time()

        if epoch % opt.test_epoch == 0:
            logging('-' * 89)
            eval_start_time = time.time()
            model.eval()
            ign_f1, ign_auc, pr_x, pr_y = test(model, dev_loader, model_name, id2rel=id2rel)
            model.train()
            logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
            logging('-' * 89)

            if ign_f1 > best_ign_f1:
                best_ign_f1 = ign_f1
                best_ign_auc = ign_auc
                best_epoch = epoch
                path = os.path.join(checkpoint_dir, model_name + '_best.pt')
                torch.save({
                    'epoch': epoch,
                    'checkpoint': model.state_dict(),
                    'lr': lr,
                    'best_ign_f1': ign_f1,
                    'best_ign_auc': ign_auc,
                    'best_epoch': epoch
                }, path)

                plt.plot(pr_x, pr_y, lw=2, label=str(epoch))
                plt.legend(loc="upper right")
                plt.savefig(os.path.join(fig_result_dir, model_name))

        if epoch % opt.save_model_freq == 0:
            path = os.path.join(checkpoint_dir, model_name + '_{}.pt'.format(epoch))
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'checkpoint': model.state_dict()
            }, path)

    print("Finish training")
    print("Best epoch = %d | Best Ign F1 = %f" % (best_epoch, best_ign_f1))
    print("Storing best result...")
    print("Finish storing")


if __name__ == '__main__':
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    opt = get_opt()
    print(json.dumps(opt.__dict__, indent=4))
    opt.data_word_vec = word2vec
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)
    train(opt)
