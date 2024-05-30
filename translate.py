import random
import re
import os
import sys

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(path)
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
import copy
import math

from utils.smiles_utils import *
from utils.translate_utils import translate_batch_original, translate_batch_stepwise
from utils.build_utils import build_model, build_iterator, load_checkpoint
from utils.model_utils import validate

from tqdm import tqdm
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw

# from rdkit.Chem.Draw import rdMolDraw2D

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:3', help='device GPU/CPU')
parser.add_argument('--batch_size_val', type=int, default=32, help='batch size')
parser.add_argument('--batch_size_trn', type=int, default=32, help='batch size')
parser.add_argument('--beam_size', type=int, default=10, help='beam size')
parser.add_argument('--stepwise', type=str, default='False', choices=['True', 'False'], help='')
parser.add_argument('--use_template', type=str, default='False', choices=['True', 'False'], help='')
parser.add_argument('--gamma', type=float, default=2.0, help='')
parser.add_argument("--seed", type=int, default=22)

# graph
parser.add_argument('--pos_enc_dim', type=int, default=20, help='')
parser.add_argument('--pe_init', type=str, default='rand_walk', help='')
parser.add_argument('--g_L', type=int, default=4, help='')
parser.add_argument('--g_hidden_dim', type=int, default=128, help='')
parser.add_argument('--g_residual', type=bool, default=True, help='')
parser.add_argument('--g_edge_feat', type=bool, default=True, help='')
parser.add_argument('--g_readout', type=str, default='mean', help='')
parser.add_argument('--g_in_feat_dropout', type=float, default=0.0, help='')
parser.add_argument('--g_dropout', type=float, default=0.3, help='')
parser.add_argument('--g_batch_norm', type=bool, default=True, help='')
parser.add_argument('--g_use_lapeig_loss', type=bool, default=False, help='')
parser.add_argument('--g_alpha_loss', type=float, default=1e-4, help='')
parser.add_argument('--g_lambda_loss', type=float, default=1, help='')

parser.add_argument('--encoder_num_layers', type=int, default=8, help='number of layers of transformer')
parser.add_argument('--decoder_num_layers', type=int, default=8, help='number of layers of transformer')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
parser.add_argument('--d_ff', type=int, default=2048, help='')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--known_class', type=str, default='True', choices=['True', 'False'],
                    help='with reaction class known/unknown')
parser.add_argument('--shared_vocab', type=str, default='False', choices=['True', 'False'],
                    help='whether sharing vocab')
parser.add_argument('--shared_encoder', type=str, default='False', choices=['True', 'False'],
                    help='whether sharing encoder')

parser.add_argument('--data_dir', type=str, default='data/uspto50k', help='base directory')
parser.add_argument('--intermediate_dir', type=str, default='intermediate_aug10', help='intermediate directory')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                    help='checkpoint directory')
# parser.add_argument('--checkpoint_dir', type=str, default='lr5e4_05contras_checkpoint',
#                     help='checkpoint directory')
# parser.add_argument('--checkpoint_dir', type=str, default='8layers_no_contras_checkpoint',
#                     help='checkpoint directory')

parser.add_argument('--checkpoint', type=str, default='a_model_1520000.pt', help='checkpoint model file')

args = parser.parse_args()


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def draw_mol_with_center(mol, idx, react_atom_index_i, react_bond_index_i=None, draw_gt=False):
    d = rdMolDraw2D.MolDraw2DCairo(500, 500)
    rdMolDraw2D.PrepareMolForDrawing(mol)
    d.DrawMolecule(mol, highlightAtoms=react_atom_index_i, highlightBonds=react_bond_index_i)
    d.FinishDrawing()
    if draw_gt:
        d.WriteDrawingText(f'./visualization/{idx}_rc_gt.png')
    else:
        d.WriteDrawingText(f'./visualization/{idx}_rc_pre.png')
    pass


def pred_reaction_center(args, batch, atom_rc_scores, bond_rc_scores, dataset, draw_gt=True):
    src, tgt, gt_context_alignment, reacts_graph, product_graph, \
        reacts_graph_num, product_graph_num, gt_atom_idx, gt_edge_idx, new_batch_adj, rea_batch_adj = batch

    batch_size = src.shape[1]

    # Atom-level reaction center accuracy:
    gt_atom_rc_label = np.array([])
    for j in range(len(product_graph_num[0])):
        l = np.array([False] * product_graph_num[0][j])
        l[gt_atom_idx[j]] = True
        gt_atom_rc_label = np.hstack((gt_atom_rc_label, l))
    pred_arc = (atom_rc_scores.squeeze(1) > 0.5).cpu().numpy().astype(np.int32).reshape(-1)
    gt_atom_rc_label = gt_atom_rc_label.astype(np.int32)
    # Bond-level reaction center accuracy:
    gt_edge_rc_label = np.array([])
    for j in range(len(product_graph_num[1])):
        l = np.array([False] * product_graph_num[1][j])
        l[gt_edge_idx[j]] = True
        gt_edge_rc_label = np.hstack((gt_edge_rc_label, l))
    pred_brc = (bond_rc_scores > 0.5).cpu().numpy().astype(np.int32).reshape(-1)

    atom_start_index = 0
    bond_start_index = 0
    for idx in range(batch_size):
        atom_index_i = pred_arc[atom_start_index:atom_start_index + product_graph_num[0][idx]].nonzero()[0]
        bond_index_i = pred_brc[bond_start_index:bond_start_index + product_graph_num[1][idx]].nonzero()[0]

        gt_atom_index_i = gt_atom_rc_label[atom_start_index:atom_start_index + product_graph_num[0][idx]].nonzero()[0]
        gt_bond_index_i = gt_edge_rc_label[bond_start_index:bond_start_index + product_graph_num[1][idx]].nonzero()[0]
        atom_start_index += product_graph_num[0][idx]
        bond_start_index += product_graph_num[1][idx]

        gt = ''.join(dataset.reconstruct_smi(src[1:, idx], src=True))
        mol = Chem.MolFromSmiles(gt)

        atom_ids = atom_index_i.tolist()
        bond_index_a = product_graph.edges()[0][bond_index_i].cpu().numpy().tolist()
        bond_index_b = product_graph.edges()[1][bond_index_i].cpu().numpy().tolist()
        bond_ids = list(set([mol.GetBondBetweenAtoms(bond_index_a[k], bond_index_b[k]).GetIdx() for k in
                             range(len(bond_index_a))]))
        draw_mol_with_center(mol, batch_size * args.batch_index + idx, atom_ids, bond_ids)

        if draw_gt:
            gt_atom_ids = gt_atom_index_i.tolist()
            gt_bond_index_a = product_graph.edges()[0][gt_bond_index_i].cpu().numpy().tolist()
            gt_bond_index_b = product_graph.edges()[1][gt_bond_index_i].cpu().numpy().tolist()
            gt_bond_ids = list(set([mol.GetBondBetweenAtoms(gt_bond_index_a[k], gt_bond_index_b[k]).GetIdx() for k in
                                    range(len(gt_bond_index_a))]))
            draw_mol_with_center(mol, batch_size * args.batch_index + idx, gt_atom_ids, gt_bond_ids, draw_gt=draw_gt)


def translate(iterator, model, dataset, vis=False):
    ground_truths = []
    generations = []
    ground_truths_am = []
    generations_am = []

    invalid_token_indices = [dataset.tgt_stoi['<unk>']]
    # Translate:
    for index, batch in enumerate(tqdm(iterator, total=len(iterator))):
        src, tgt, _, _, _, _, _, _, _, _, _, reac_smi = batch
        args.batch_index = index
        if args.stepwise == 'False':
            # Original Main:
            pred_tokens, pred_scores, atom_rc_scores, bond_rc_scores = translate_batch_original(args, model, batch,
                                                                                                beam_size=args.beam_size,
                                                                                                invalid_token_indices=invalid_token_indices)
            if vis:
                pred_reaction_center(args, batch, atom_rc_scores, bond_rc_scores, dataset)
            for idx in range(batch[0].shape[1]):
                # gt = ''.join(dataset.reconstruct_smi(tgt[:, idx], src=False))
                # gt_am = reac_smi[idx]
                # hypos_am = np.array(
                #     [''.join(dataset.reconstruct_smi(tokens, src=False)) for tokens in pred_tokens[idx]])
                # hypo_am_len = np.array([len(smi_tokenizer(ht)) for ht in hypos_am])
                # new_pred_am_score = copy.deepcopy(pred_scores[idx]).cpu().numpy() / hypo_am_len
                # ordering_am = np.argsort(new_pred_am_score)[::-1]

                gt = remove_am_without_canonical(reac_smi[idx])
                hypos = np.array(
                    [remove_am_without_canonical(''.join(dataset.reconstruct_smi(tokens, src=False))) for tokens in
                     pred_tokens[idx]])
                hypo_len = np.array([len(smi_tokenizer(ht)) for ht in hypos])
                new_pred_score = copy.deepcopy(pred_scores[idx]).cpu().numpy() / hypo_len
                ordering = np.argsort(new_pred_score)[::-1]

                ground_truths.append(gt)
                generations.append(hypos[ordering])

                # ground_truths_am.append(gt_am)
                # generations_am.append(hypos_am[ordering_am])

                if vis:
                    try:
                        gt = canonical_smiles(gt)
                        top_pre = canonical_smiles(generations[-1][0])
                        mol = Chem.MolFromSmiles(gt)
                        Draw.MolToFile(mol,
                                       f'./visualization/{args.batch_size_val * index + idx}_reactants_gt.png')
                        mol = Chem.MolFromSmiles(top_pre)
                        Draw.MolToFile(mol,
                                       f'./visualization/{args.batch_size_val * index + idx}_reactants_{gt == top_pre}.png')
                    except:
                        continue

        else:
            # Stepwise Main:
            # untyped: T=10; beta=0.5, percent_aa=40, percent_ab=40
            # typed: T=10; beta=0.5, percent_aa=40, percent_ab=55
            if args.known_class == 'True':
                percent_ab = 55
            else:
                percent_ab = 40
            pred_tokens, pred_scores, predicts = \
                translate_batch_stepwise(model, batch, beam_size=args.beam_size,
                                         invalid_token_indices=invalid_token_indices,
                                         T=10, alpha_atom=-1, alpha_bond=-1,
                                         beta=0.5, percent_aa=40, percent_ab=percent_ab, k=3,
                                         use_template=args.use_template == 'True',
                                         factor_func=dataset.factor_func,
                                         reconstruct_func=dataset.reconstruct_smi,
                                         rc_path=args.intermediate_dir + '/rt2reaction_center.pk')

            original_beam_size = pred_tokens.shape[1]
            current_i = 0
            for batch_i, predict in enumerate(predicts):
                gt = ''.join(dataset.reconstruct_smi(tgt[:, batch_i], src=False))
                remain = original_beam_size
                beam_size = math.ceil(original_beam_size / len(predict))

                # normalized_reaction_center_score = np.array([pred[1] for pred in predict]) / 10
                hypo_i, hypo_scores_i = [], []
                for j, (rc, rc_score) in enumerate(predict):
                    # rc_score = normalized_reaction_center_score[j]

                    pred_token = pred_tokens[current_i + j]

                    sub_hypo_candidates, sub_score_candidates = [], []
                    for k in range(pred_token.shape[0]):
                        hypo_smiles_k = ''.join(dataset.reconstruct_smi(pred_token[k], src=False))
                        hypo_lens_k = len(smi_tokenizer(hypo_smiles_k))
                        hypo_scores_k = pred_scores[current_i + j][k].cpu().numpy() / hypo_lens_k + rc_score / 10

                        if hypo_smiles_k not in hypo_i:  # only select unique entries
                            sub_hypo_candidates.append(hypo_smiles_k)
                            sub_score_candidates.append(hypo_scores_k)

                    ordering = np.argsort(sub_score_candidates)[::-1]
                    sub_hypo_candidates = list(np.array(sub_hypo_candidates)[ordering])[:min(beam_size, remain)]
                    sub_score_candidates = list(np.array(sub_score_candidates)[ordering])[:min(beam_size, remain)]

                    hypo_i += sub_hypo_candidates
                    hypo_scores_i += sub_score_candidates

                    remain -= beam_size

                current_i += len(predict)
                ordering = np.argsort(hypo_scores_i)[::-1][:args.beam_size]
                ground_truths.append(gt)
                generations.append(np.array(hypo_i)[ordering])

    return ground_truths, generations, ground_truths_am, generations_am


def main(args):
    set_seed(args.seed)
    # Build Data Iterator:
    iterator, dataset = build_iterator(args, train=False)
    epoch = int(args.checkpoint.split('_')[-1].split('.')[0])
    # args.checkpoint = f'a_model_{epoch}.pt'
    # Load Checkpoint Model:
    model = build_model(args, dataset.src_itos, dataset.tgt_itos)
    _, _, model = load_checkpoint(args, model)
    # Get Output Path:
    dec_version = 'stepwise' if args.stepwise == 'True' else 'vanilla'
    exp_version = 'typed' if args.known_class == 'True' else 'untyped'
    aug_version = '_augment' if 'augment' in args.checkpoint_dir else ''
    tpl_version = '_template' if args.use_template == 'True' else ''
    file_name = 'result/{}_bs_top{}_generation_{}{}{}_epoch_{}.pk'.format(dec_version, args.beam_size, exp_version,
                                                                 aug_version, tpl_version,epoch)
    output_path = os.path.join(args.intermediate_dir, file_name)
    print('Output path: {}'.format(output_path))

    # Begin Translating:
    ground_truths, generations, ground_truths_am, generations_am = translate(iterator, model, dataset)
    accuracy_matrix = np.zeros((len(ground_truths), args.beam_size))
    for i in range(len(ground_truths)):
        gt_i = canonical_smiles(ground_truths[i])
        generation_i = [canonical_smiles(gen) for gen in generations[i] if canonical_smiles(gen) != None]
        generation_i += [None] * (args.beam_size - len(generation_i))
        for j in range(args.beam_size):
            if gt_i in generation_i[:j + 1]:
                accuracy_matrix[i][j] = 1


    # for j in range(args.beam_size):
    #     print('Top-{}: {}'.format(j + 1, round(np.mean(accuracy_matrix[:, j]), 4)))
    for j in range(args.beam_size):
        print('{}'.format(round(np.mean(accuracy_matrix[:, j]), 4)))
    with open(output_path, 'wb') as f:
        pickle.dump((ground_truths, generations), f)

    accuracy_arc, accuracy_brc, accuracy_token = \
        validate(args, model, iterator, model.embedding_tgt.word_padding_idx)
    print_line = 'Validation accuracy: \n {} \n {} \n {}'.format(round(accuracy_arc, 4),
                                                                 round(accuracy_brc, 4),
                                                                 round(accuracy_token, 4))
    print(print_line)
    return


if __name__ == "__main__":
    print(args)
    with open('args.pk', 'wb') as f:
        pickle.dump(args, f)
    if args.known_class == 'True':
        args.checkpoint_dir = args.checkpoint_dir + '_typed'
    else:
        args.checkpoint_dir = args.checkpoint_dir + '_untyped'
    args.proj_path = path
    print(f"os.getcwd:{args.proj_path}")
    args.checkpoint_dir = os.path.join(args.proj_path, args.checkpoint_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.intermediate_dir = os.path.join(args.proj_path, args.intermediate_dir)
    if not os.path.exists(args.intermediate_dir):
        os.makedirs(args.intermediate_dir, exist_ok=True)
    args.data_dir = os.path.join('/'.join(args.proj_path.split('/')), args.data_dir)
    main(args)
