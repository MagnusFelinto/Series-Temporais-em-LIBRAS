"""
montar_dataset.py — Etapa 2 do plano: transforma os keypoints (kp\\*.npz) em um dataset temporal rotulado.

Entrada : D:\\ProjetoMestrado\\data\\malta_35\\kp\\registry.csv + kp\\<sinal>\\<arquivo>.npz   (saída da Etapa 1)
          D:\\ProjetoMestrado\\data\\malta_35\\classes.csv                                    (sinal -> label)
Saída   : D:\\ProjetoMestrado\\data\\malta_35\\dataset_kp_v1.npz
          D:\\ProjetoMestrado\\data\\malta_35\\dataset_kp_v1_index.csv   (uma linha por amostra)

Conteúdo do .npz (N = número de vídeos, T = 32 frames):
    X_A, X_B, X_C  float32 (N, T, F)  três conjuntos de colunas (A = 201, B = 321, C = 465)
    cols_A/B/C     str     (F,)       nome de cada coluna
    mask           bool    (N, T, 4)  frame tem [pose, mão dominante, outra mão, rosto] (real ou interpolado)
    y              int64   (N,)       rótulo (classes.csv)
    sinal, familia, dictionary, arquivo  str (N,)
    classes        str     (K,)       nome de cada rótulo, na ordem do índice

Tratamentos, nesta ordem (ver o plano, Etapa 2):
    1. coordenadas em pixels do frame processado (corrige a proporção)
    2. interpolação linear de falhas curtas (<= 4 frames); falhas maiores ficam vazias e marcadas na máscara
    3. suavização por média móvel de 3 frames
    4. corte do repouso (início/fim parados) e reamostragem de volta para 32 frames
    5. normalização do corpo: centro = ponto médio dos ombros (mediana do vídeo);
       x e z divididos pela largura dos ombros, y pela distância vertical ombros–nariz (por eixo,
       o que neutraliza diferenças de proporção entre fontes)
    6. padronização da mão dominante (espelha e troca os lados quando a esquerda se move mais)
    7. forma da mão: cada mão relativa ao próprio pulso, dividida pelo tamanho da mão
    8. velocidades (Δx, Δy) dos pulsos e das pontas dos dedos
Valores ausentes viram 0 depois da normalização; a máscara diz o que é real.

Requisitos: numpy, pandas (não precisa de MediaPipe nem PyTorch).
Uso: python montar_dataset.py --raiz D:\\ProjetoMestrado\\data\\malta_35
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

T = 32
POSE_IDX = list(range(25))                       # pose sem as pernas
POSE_PARES = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
              (17, 18), (19, 20), (21, 22), (23, 24)]
# subconjunto do rosto: pares simétricos (esquerda, direita) + pontos centrais
FACE_PARES = [(33, 263), (133, 362), (159, 386), (145, 374), (70, 300), (105, 334), (107, 336), (46, 276),
              (61, 291), (37, 267), (84, 314), (40, 270), (91, 321), (234, 454), (172, 397), (136, 365)]
FACE_CENTRO = [1, 2, 0, 13, 14, 17, 152, 10]
FACE_IDX = [i for par in FACE_PARES for i in par] + FACE_CENTRO      # 40 pontos
PONTAS = [4, 8, 12, 16, 20]


# ----------------------------------------------------------------------------- utilidades temporais
def interpolar_falhas(a, max_gap=4):
    """a: (T, P, D) com NaN. Interpola linearmente falhas internas de até max_gap frames.
    Devolve o array preenchido e a máscara (T,) de frames válidos (reais ou interpolados)."""
    a = a.copy()
    valido = ~np.isnan(a[:, 0, 0])
    idx = np.where(valido)[0]
    if len(idx) == 0:
        return a, valido
    for k in range(len(idx) - 1):
        i0, i1 = idx[k], idx[k + 1]
        gap = i1 - i0 - 1
        if 0 < gap <= max_gap:
            for j in range(1, gap + 1):
                w = j / (gap + 1)
                a[i0 + j] = (1 - w) * a[i0] + w * a[i1]
    return a, ~np.isnan(a[:, 0, 0])


def preencher_bordas(a):
    """Para pose e rosto (quase sempre presentes): preenche frames vazios com o vizinho válido mais próximo."""
    a = a.copy()
    valido = ~np.isnan(a[:, 0, 0])
    if valido.all() or not valido.any():
        return a
    idx = np.where(valido)[0]
    for t in np.where(~valido)[0]:
        a[t] = a[idx[np.argmin(np.abs(idx - t))]]
    return a


def suavizar(a, k=3):
    """Média móvel centrada de k frames, ignorando NaN (não cria valores onde não havia)."""
    out = a.copy()
    h = k // 2
    for t in range(len(a)):
        if np.isnan(a[t, 0, 0]):
            continue
        janela = a[max(0, t - h): t + h + 1]
        out[t] = np.nanmean(janela, axis=0)
    return out


def reamostrar(a, t0, t1, n=T):
    """Reamostra o trecho [t0, t1] de a (T, ...) para n frames por interpolação linear.
    Onde um dos vizinhos é NaN, usa o vizinho mais próximo (NaN se ele também for)."""
    pos = np.linspace(t0, t1, n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(a) - 1)
    w = (pos - lo).reshape((-1,) + (1,) * (a.ndim - 1))
    A, B = a[lo], a[hi]
    out = (1 - w) * A + w * B
    perto = np.where(w < 0.5, A, B)
    ruim = np.isnan(out)
    out[ruim] = perto[ruim]
    return out


def tamanho_mao(m):
    return np.linalg.norm(m[:, 9, :2] - m[:, 0, :2], axis=1)          # pulso -> base do dedo médio


def limpar_mao(m, fator=2.0):
    """Descarta frames em que o tamanho da mão foge muito da mediana do vídeo (detecção ruim)."""
    m = m.copy()
    tam = tamanho_mao(m)
    med = np.nanmedian(tam) if np.isfinite(tam).any() else np.nan
    if np.isfinite(med) and med > 0:
        ruim = (tam < med / fator) | (tam > med * fator)
        m[ruim] = np.nan
    return m


# ----------------------------------------------------------------------------- processamento de um vídeo
def processar(d, margem=2, limiar_rep=0.25, min_frames=10, k_suavizar=3):
    W, H = d["frame_wh"].astype(np.float32)
    escala = np.array([W, H, W], np.float32)
    pose = d["pose"][:, POSE_IDX, :3] * escala
    vis = d["pose"][:, POSE_IDX, 3]
    lh = d["lh"] * escala
    rh = d["rh"] * escala
    face = d["face"][:, FACE_IDX, :] * escala

    # 2. falhas
    pose = preencher_bordas(pose)
    face = preencher_bordas(face)
    lh, rh = limpar_mao(lh), limpar_mao(rh)
    lh, _ = interpolar_falhas(lh)
    rh, _ = interpolar_falhas(rh)
    # 3. suavização
    if k_suavizar > 1:   # k_suavizar=1 desliga a suavização (versão bruta, para o estudo de ablação)
        pose, face, lh, rh = (suavizar(pose, k_suavizar), suavizar(face, k_suavizar),
                              suavizar(lh, k_suavizar), suavizar(rh, k_suavizar))

    # referências do corpo (mediana do vídeo)
    ombro_e, ombro_d, nariz = pose[:, 11], pose[:, 12], pose[:, 0]
    centro = np.nanmedian((ombro_e + ombro_d) / 2, axis=0)
    sx = np.nanmedian(np.abs(ombro_e[:, 0] - ombro_d[:, 0]))
    sy = np.nanmedian(np.abs((ombro_e[:, 1] + ombro_d[:, 1]) / 2 - nariz[:, 1]))
    if not np.isfinite(sx) or sx <= 1 or not np.isfinite(sy) or sy <= 1:
        raise ValueError("pose sem ombros/nariz confiáveis")
    norm = np.array([sx, sy, sx], np.float32)

    # 4. corte do repouso: um frame é "ativo" quando algum pulso está longe tanto da posição do
    #    primeiro quanto do último frame (posições de repouso), em larguras de ombro
    pulsos = pose[:, [15, 16], :2] / norm[:2]
    longe = np.minimum(np.linalg.norm(pulsos - pulsos[:1], axis=2),
                       np.linalg.norm(pulsos - pulsos[-1:], axis=2)).max(axis=1)      # (T,)
    ativos = np.where(longe > limiar_rep)[0]
    if len(ativos) > 0:
        t0 = max(0, ativos[0] - margem)
        t1 = min(T - 1, ativos[-1] + margem)
    else:
        t0, t1 = 0, T - 1
    if t1 - t0 + 1 < min_frames:                 # trecho curto demais: centraliza uma janela mínima
        meio = (t0 + t1) / 2
        t0 = int(max(0, round(meio - min_frames / 2)))
        t1 = int(min(T - 1, t0 + min_frames - 1))

    # mão dominante (decidida antes do corte): a mão detectada e levantada acima do meio do tronco
    # em mais frames; o caminho percorrido só desempata
    meio_tronco = (pose[:, 11, 1] + pose[:, 12, 1] + pose[:, 23, 1] + pose[:, 24, 1]) / 4
    ativa_e = int((~np.isnan(lh[:, 0, 0]) & (pose[:, 15, 1] < meio_tronco)).sum())
    ativa_d = int((~np.isnan(rh[:, 0, 0]) & (pose[:, 16, 1] < meio_tronco)).sum())
    caminho = np.linalg.norm(np.diff(pulsos, axis=0), axis=2).sum(axis=0)
    if ativa_e != ativa_d:
        espelhado = bool(ativa_e > 1.2 * ativa_d and ativa_e >= 3)
    else:
        espelhado = bool(caminho[0] > 1.2 * caminho[1])

    pose, face = reamostrar(pose, t0, t1), reamostrar(face, t0, t1)
    lh, rh = reamostrar(lh, t0, t1), reamostrar(rh, t0, t1)

    # 5. normalização
    pose, face = (pose - centro) / norm, (face - centro) / norm
    lh, rh = (lh - centro) / norm, (rh - centro) / norm

    # 6. espelhamento (decisão tomada antes do corte)
    if espelhado:
        for arr in (pose, face, lh, rh):
            arr[..., 0] *= -1
        for a, b in POSE_PARES:
            pose[:, [a, b]] = pose[:, [b, a]]
        n = len(FACE_PARES)
        ordem = list(range(len(FACE_IDX)))
        for k in range(n):
            ordem[2 * k], ordem[2 * k + 1] = 2 * k + 1, 2 * k
        face = face[:, ordem]
        lh, rh = rh, lh
    dom, outra = rh, lh                            # depois disso, a mão dominante fica sempre na coluna "direita"

    mask = np.stack([~np.isnan(pose[:, 0, 0]), ~np.isnan(dom[:, 0, 0]),
                     ~np.isnan(outra[:, 0, 0]), ~np.isnan(face[:, 0, 0])], axis=1)

    # 7. forma da mão (relativa ao pulso, escala = pulso -> base do dedo médio)
    def forma(m):
        rel = m - m[:, :1]
        tam = np.nanmedian(tamanho_mao(m)) if np.isfinite(tamanho_mao(m)).any() else np.nan
        if not np.isfinite(tam) or tam <= 1e-6:
            return np.full_like(rel[:, 1:], np.nan)
        return rel[:, 1:] / tam                   # 20 pontos (o pulso vira sempre 0), escala da mediana do vídeo
    f_dom, f_outra = forma(dom), forma(outra)

    # 8. velocidades dos pulsos e pontas dos dedos (x, y)
    pts = np.concatenate([pose[:, [15, 16], :2], dom[:, PONTAS, :2], outra[:, PONTAS, :2]], axis=1)  # (T, 12, 2)
    vel_pts = np.diff(pts, axis=0, prepend=pts[:1])

    blocos = dict(pose=pose, dom=dom, outra=outra, f_dom=f_dom, f_outra=f_outra, face=face, vel=vel_pts)
    blocos = {k: np.nan_to_num(v.reshape(T, -1), nan=0.0).astype(np.float32) for k, v in blocos.items()}
    info = dict(t0=int(t0), t1=int(t1), espelhado=espelhado, ativa_e=ativa_e, ativa_d=ativa_d, sx_px=float(sx), sy_px=float(sy),
                dom_pct=float(mask[:, 1].mean()), outra_pct=float(mask[:, 2].mean()))
    return blocos, mask, info


def nomes_colunas():
    eixos3, eixos2 = ["x", "y", "z"], ["x", "y"]
    pose = [f"pose{i}_{e}" for i in POSE_IDX for e in eixos3]
    dom = [f"dom{i}_{e}" for i in range(21) for e in eixos3]
    outra = [f"outra{i}_{e}" for i in range(21) for e in eixos3]
    f_dom = [f"fdom{i}_{e}" for i in range(1, 21) for e in eixos3]
    f_outra = [f"foutra{i}_{e}" for i in range(1, 21) for e in eixos3]
    face = [f"face{i}_{e}" for i in FACE_IDX for e in eixos3]
    vel = [f"v_{p}_{e}" for p in ["pulsoE", "pulsoD"] + [f"dom{i}" for i in PONTAS] + [f"outra{i}" for i in PONTAS]
           for e in eixos2]
    return dict(pose=pose, dom=dom, outra=outra, f_dom=f_dom, f_outra=f_outra, face=face, vel=vel)


VERSOES = {"A": ["pose", "dom", "outra"],
           "B": ["pose", "dom", "outra", "f_dom", "f_outra"],
           "C": ["pose", "dom", "outra", "f_dom", "f_outra", "face", "vel"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=r"D:\ProjetoMestrado\data\malta_35")
    ap.add_argument("--kp", default=None, help="padrão: <raiz>\\kp")
    ap.add_argument("--saida", default=None, help="padrão: <raiz>\\dataset_kp_v1.npz")
    ap.add_argument("--k_suavizar", type=int, default=3, help="janela da média móvel; 1 = sem suavização")
    ap.add_argument("--so_familia", default=None, help="usa só as famílias que começam com este prefixo (ex.: MINDS_)")
    a = ap.parse_args()
    raiz = Path(a.raiz)
    kp = Path(a.kp) if a.kp else raiz / "kp"
    saida = Path(a.saida) if a.saida else raiz / "dataset_kp_v1.npz"

    classes = pd.read_csv(raiz / "classes.csv", encoding="utf-8-sig").sort_values("label")
    reg = pd.read_csv(kp / "registry.csv", encoding="utf-8-sig")
    reg = reg[reg.erro.fillna("") == ""].sort_values("arquivo").reset_index(drop=True)
    if a.so_familia:
        reg = reg[reg.familia.astype(str).str.startswith(a.so_familia)].reset_index(drop=True)
    cols = nomes_colunas()

    Xs = {v: [] for v in VERSOES}
    masks, linhas = [], []
    for r in reg.itertuples():
        d = np.load(kp / r.npz, allow_pickle=False)
        try:
            blocos, mask, info = processar(d, k_suavizar=a.k_suavizar)
        except ValueError as e:
            print("pulado:", r.arquivo, e)
            continue
        for v, partes in VERSOES.items():
            Xs[v].append(np.concatenate([blocos[p] for p in partes], axis=1))
        masks.append(mask)
        linhas.append(dict(i=len(linhas), arquivo=r.arquivo, sinal=r.sinal, y=int(r.label), familia=r.familia,
                           dictionary=r.dictionary, **info))

    idx = pd.DataFrame(linhas)
    out = {f"X_{v}": np.stack(Xs[v]) for v in VERSOES}
    out.update({f"cols_{v}": np.array(sum((cols[p] for p in partes), [])) for v, partes in VERSOES.items()})
    out.update(mask=np.stack(masks), y=idx.y.to_numpy(np.int64), sinal=idx.sinal.to_numpy(str),
               familia=idx.familia.to_numpy(str), dictionary=idx.dictionary.to_numpy(str),
               arquivo=idx.arquivo.to_numpy(str), classes=classes.sinal.to_numpy(str))
    saida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(saida, **out)
    idx.to_csv(saida.with_name(saida.stem + "_index.csv"), index=False, encoding="utf-8-sig")

    for v in VERSOES:
        X = out[f"X_{v}"]
        print(f"X_{v}: {X.shape}  colunas={len(out[f'cols_{v}'])}  NaN={np.isnan(X).sum()}  "
              f"faixa=[{X.min():.2f}, {X.max():.2f}]")
    print(f"amostras={len(idx)}  classes={idx.y.nunique()}  espelhados={idx.espelhado.sum()}  "
          f"frames mantidos (mediana)={int((idx.t1 - idx.t0 + 1).median())}")
    print(f"salvo em {saida}")


if __name__ == "__main__":
    main()
