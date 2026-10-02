"""
Implementação própria de IDF1, ID switches e fragmentações
"""

import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    """Sobreposicao (Intersection-over-Union) entre duas caixas (x,y,w,h)"""
    ax2, ay2 = a['x'] + a['w'], a['y'] + a['h']
    bx2, by2 = b['x'] + b['w'], b['y'] + b['h']
    inter = (max(0.0, min(ax2, bx2) - max(a['x'], b['x'])) *
             max(0.0, min(ay2, by2) - max(a['y'], b['y'])))
    
    uniao = a['w'] * a['h'] + b['w'] * b['h'] - inter
    return inter / uniao if uniao > 0 else 0.0


def agrupa_por_frame(itens):
    """Indexa uma lista de deteccoes/anotacoes por numero de quadro"""
    d = defaultdict(list)
    for it in itens:
        d[it['frame']].append(it)
        
    return d


def _casa_quadro(G, P, iou_thr):
    """Casa as caixas de UM quadro por IoU (Hungarian). Devolve pares (id_gt, id_pred)"""
    if not G or not P:
        return []
    
    M = np.zeros((len(G), len(P)))
    for a, g in enumerate(G):
        for b, p in enumerate(P):
            M[a, b] = iou(g, p)

    lin, col = linear_sum_assignment(-M)
    return [(G[i]['id'], P[j]['id']) for i, j in zip(lin, col) if M[i, j] >= iou_thr]


def idf1(gt, pred, iou_thr=0.5):
    """IDF1: exige uma atribuicao global 1 para 1 entre identidades previstas e verdadeiras ao longo da sequencia inteira

    O denominador 2*IDTP + IDFP + IDFN simplifica para total_gt + total_pred
    (constante), logo maximizar IDF1 equivale a maximizar IDTP e por isso o
    Hungarian (linear_sum_assignment) resolve a atribuicao otima de uma vez.
    """
    gt_ids = sorted(set(r['id'] for r in gt))
    pr_ids = sorted(set(r['id'] for r in pred))
    G = {i: {} for i in gt_ids}
    [G[r['id']].__setitem__(r['frame'], r) for r in gt]
    P = {j: {} for j in pr_ids}
    [P[r['id']].__setitem__(r['frame'], r) for r in pred]
    total_gt, total_pr = len(gt), len(pred)

    # quantos quadros cada id verdadeiro e cada id previsto casam
    W = np.zeros((len(gt_ids), len(pr_ids)))
    for a, i in enumerate(gt_ids):
        for b, j in enumerate(pr_ids):
            comuns = set(G[i]) & set(P[j])
            W[a, b] = sum(1 for f in comuns if iou(G[i][f], P[j][f]) >= iou_thr)

    # casamento 1 para 1 que maximiza IDTP
    n = max(len(gt_ids), len(pr_ids), 1)
    C = np.zeros((n, n))
    C[:len(gt_ids), :len(pr_ids)] = W
    lin, col = linear_sum_assignment(-C)
    IDTP = float(sum(C[r, c] for r, c in zip(lin, col)))

    # (3) precisao/recall de identidade + F1
    IDFN, IDFP = total_gt - IDTP, total_pr - IDTP
    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN) if (2 * IDTP + IDFP + IDFN) > 0 else 0.0
    return dict(IDF1=IDF1, IDTP=int(IDTP), IDFP=int(IDFP), IDFN=int(IDFN))


def contar_id_switches(gt, pred, iou_thr=0.5):
    """Numero de trocas de identidade"""
    G, P = agrupa_por_frame(gt), agrupa_por_frame(pred)
    ultimo_pred, switches = {}, 0
    for t in sorted(set(G) | set(P)):
        for gid, pid in _casa_quadro(G.get(t, []), P.get(t, []), iou_thr):
            if gid in ultimo_pred and ultimo_pred[gid] != pid:
                switches += 1

            ultimo_pred[gid] = pid

    return switches


def contar_fragmentacoes(gt, pred, iou_thr=0.5):
    """Numero de interrupcoes de uma track verdadeira.

    Fragmentacao = a GT estava casada, deixou de casar, e voltou a casar.
    Diferente do switch: aqui não importa se o id previsto mudou, o que importa é a
    interrupcao da continuidade. Uma track pode fragmentar sem switch (sumiu e
    voltou com o mesmo id) e trocar de id sem fragmentar (nunca chegou a sumir).
    """
    G, P = agrupa_por_frame(gt), agrupa_por_frame(pred)
    frames = sorted(set(G) | set(P))
    estado = {}   
    frags = 0
    for t in frames:
        pares = _casa_quadro(G.get(t, []), P.get(t, []), iou_thr)
        casados_agora = set(gid for gid, pid in pares)
        gids_presentes = set(r['id'] for r in G.get(t, []))
        for gid in gids_presentes:
            st = estado.get(gid, 'nunca')
            if gid in casados_agora:
                if st == 'perdido':      
                    frags += 1
                estado[gid] = 'ativo'
            else:
                if st == 'ativo':        
                    estado[gid] = 'perdido'
    return frags


def average_precision(gt, dets, iou_thr=0.5):
    """mAP de deteccao: area sob a curva precisao-recall"""
    n_gt = len(gt)
    if n_gt == 0:
        return 0.0
    
    gt_f = agrupa_por_frame(gt)
    usados = {f: [False] * len(v) for f, v in gt_f.items()}
    D = sorted(dets, key=lambda d: -d.get('conf', 1.0))
    tp, fp = np.zeros(len(D)), np.zeros(len(D))

    for i, d in enumerate(D):
        melhor_iou, melhor_j = 0.0, -1
        for j, g in enumerate(gt_f.get(d['frame'], [])):
            v = iou(d, g)
            if v > melhor_iou:
                melhor_iou, melhor_j = v, j
        if melhor_iou >= iou_thr and not usados[d['frame']][melhor_j]:
            tp[i] = 1
            usados[d['frame']][melhor_j] = True
        else:
            fp[i] = 1

    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall, precision = tp_c / n_gt, tp_c / np.maximum(tp_c + fp_c, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]

    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def erro_contagem_ids(gt, pred):
    """Erro de contagem de identidades unicas"""
    n_true = len(set(r['id'] for r in gt))
    n_pred = len(set(r['id'] for r in pred))
    return dict(verdadeiras=n_true, 
                previstas=n_pred,
                erro_abs=abs(n_pred - n_true),
                razao=n_pred / n_true if n_true else 0.0)
