"""
extrair_keypoints.py — Etapa 1 do plano: keypoints MediaPipe para cada vídeo do DatasetLibras.

Entrada : D:\\ProjetoMestrado\\data\\malta_35\\tensores\\<sinal>\\<arquivo>.pt  (uint8, 32 x 3 x 224 x 224, lido SEM PyTorch)
Saída   : D:\\ProjetoMestrado\\data\\malta_35\\kp\\<sinal>\\<arquivo>.npz
          D:\\ProjetoMestrado\\data\\malta_35\\kp\\registry.csv   (qualidade por vídeo)

Cada .npz contém (T = 32 frames; NaN = não detectado):
    pose      float32 (T, 33, 4)   x, y, z, visibilidade   (x, y normalizados 0–1 no frame processado)
    lh, rh    float32 (T, 21, 3)   mão esquerda / direita DA PESSOA (ligadas aos pulsos 15 / 16 da pose)
    face      float32 (T, 468, 3)
    hand_src  int8    (T, 2)       0 = ausente, 1 = Holistic, 2 = plano B (recorte no pulso)   [lh, rh]
    frame_wh  int32   (2,)         largura, altura do frame processado (para voltar a pixels)
    sinal, arquivo, dictionary, familia  (texto)

Requisitos: mediapipe==0.10.14 (traz os modelos embutidos; a API mp.solutions foi removida
nas versões 1.x), numpy, opencv-python, pandas.

Uso:
    python extrair_keypoints.py --raiz D:\\ProjetoMestrado\\data\\malta_35
    python extrair_keypoints.py --raiz D:\\ProjetoMestrado\\data\\malta_35 --limite 2   # 2 vídeos por dicionário (teste)
Pode ser interrompido e rodado de novo: vídeos com .npz pronto são pulados (use --refazer para refazer).
"""
import argparse
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import mediapipe as mp

T, S = 32, 224
POSE_L_WRIST, POSE_R_WRIST, POSE_L_ELBOW, POSE_R_ELBOW = 15, 16, 13, 14
POSE_L_SHOULDER, POSE_R_SHOULDER, POSE_L_HIP, POSE_R_HIP = 11, 12, 23, 24


# ----------------------------------------------------------------------------- leitura
def ler_tensor(caminho: Path) -> np.ndarray:
    """Lê o .pt (zip do torch.save) direto como numpy: (32, 224, 224, 3) RGB uint8."""
    with zipfile.ZipFile(caminho) as z:
        nome = next(n for n in z.namelist() if n.endswith("data/0"))
        bruto = np.frombuffer(z.read(nome), dtype=np.uint8)
    if bruto.size != T * 3 * S * S:
        raise ValueError(f"tamanho inesperado {bruto.size} em {caminho}")
    return bruto.reshape(T, 3, S, S).transpose(0, 2, 3, 1).copy()


def cortar_faixas_pretas(video: np.ndarray, limiar: float = 12.0) -> np.ndarray:
    """Remove colunas laterais quase pretas (pillarbox de vídeos do YouTube)."""
    col = video.mean(axis=(0, 1, 3))
    claras = np.where(col >= limiar)[0]
    if len(claras) == 0:
        return video
    a, b = claras[0], claras[-1] + 1
    return video[:, :, a:b] if (b - a) >= S * 0.6 else video


def preparar(video: np.ndarray, aspecto: float, escala: int) -> np.ndarray:
    """Desachata para a proporção original (largura = altura × aspecto) e amplia."""
    h = S * escala
    w = int(round(S * aspecto)) * escala
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC) for f in video])


# ----------------------------------------------------------------------------- conversões
def lm_array(lms, n, dims):
    out = np.full((n, dims), np.nan, np.float32)
    if lms is None:
        return out
    for i, p in enumerate(lms.landmark):
        out[i, :3] = (p.x, p.y, p.z)
        if dims == 4:
            out[i, 3] = p.visibility
    return out


# ----------------------------------------------------------------------------- plano B
def plano_b_mao(frame, pose, lado, detector):
    """Recorta uma janela em volta do pulso e roda o detector de mãos só nela.
    Tenta duas janelas (1,6x e 2,4x o antebraço). Aceita a mão cujo pulso fica a menos de
    0,8 antebraço do pulso da pose, para não pegar o rosto ou a outra mão."""
    ip, ic = (POSE_L_WRIST, POSE_L_ELBOW) if lado == "L" else (POSE_R_WRIST, POSE_R_ELBOW)
    H, W = frame.shape[:2]
    if np.isnan(pose[ip, 0]) or pose[ip, 3] < 0.3:
        return None
    pw = np.array([pose[ip, 0] * W, pose[ip, 1] * H])
    pc = np.array([pose[ic, 0] * W, pose[ic, 1] * H])
    antebraco = float(np.linalg.norm(pw - pc))
    if not np.isfinite(antebraco) or antebraco < 4:
        return None
    centro = pw + 0.35 * (pw - pc)               # a mão fica além do pulso, na direção do antebraço
    for fator in (1.6, 2.4):
        L = int(round(max(fator * antebraco, 64.0)))
        x0i, y0i = int(round(centro[0] - L / 2)), int(round(centro[1] - L / 2))
        img = cv2.copyMakeBorder(frame, L, L, L, L, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        crop = img[y0i + L:y0i + 2 * L, x0i + L:x0i + 2 * L]
        if crop.size == 0:
            continue
        r = detector.process(cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC))
        if not r.multi_hand_landmarks:
            continue
        melhor, dmin = None, 1e9
        for hl in r.multi_hand_landmarks:
            a = lm_array(hl, 21, 3)
            a[:, 0] = (a[:, 0] * L + x0i) / W        # volta para coordenadas do frame inteiro
            a[:, 1] = (a[:, 1] * L + y0i) / H
            a[:, 2] = a[:, 2] * L / W
            d = np.hypot((a[0, 0] - pose[ip, 0]) * W, (a[0, 1] - pose[ip, 1]) * H)
            if d < dmin:
                melhor, dmin = a, d
        if dmin < 0.8 * antebraco:
            return melhor
    return None


# ----------------------------------------------------------------------------- qualidade
def cobertura_ativa(pose, mao, ip):
    """% de frames com a mão detectada entre os frames em que o pulso está acima do meio do tronco."""
    ombro_y = np.nanmean(pose[:, [POSE_L_SHOULDER, POSE_R_SHOULDER], 1], axis=1)
    quadril_y = np.nanmean(pose[:, [POSE_L_HIP, POSE_R_HIP], 1], axis=1)
    meio = (ombro_y + quadril_y) / 2
    ativo = (pose[:, ip, 3] > 0.5) & (pose[:, ip, 1] < meio)
    if ativo.sum() == 0:
        return np.nan, 0
    return float((~np.isnan(mao[ativo, 0, 0])).mean()), int(ativo.sum())


# ----------------------------------------------------------------------------- principal
def corrigir_maos_duplicadas(pose, lh, rh, src, W, H):
    """Quando as duas mãos saem quase no mesmo lugar mas os pulsos da pose estão separados,
    a mesma mão foi atribuída aos dois lados. Mantém só a que está mais perto do próprio pulso.
    Devolve o número de frames corrigidos."""
    ombros = np.hypot((pose[:, 11, 0] - pose[:, 12, 0]) * W, (pose[:, 11, 1] - pose[:, 12, 1]) * H)
    sw = np.nanmedian(ombros) if np.isfinite(ombros).any() else np.nan
    if not np.isfinite(sw) or sw <= 0:
        return 0
    n = 0
    for t in range(len(pose)):
        if np.isnan(lh[t, 0, 0]) or np.isnan(rh[t, 0, 0]):
            continue
        d_maos = np.hypot((lh[t, :, 0] - rh[t, :, 0]) * W, (lh[t, :, 1] - rh[t, :, 1]) * H).mean()
        d_pulsos = np.hypot((pose[t, 15, 0] - pose[t, 16, 0]) * W, (pose[t, 15, 1] - pose[t, 16, 1]) * H)
        if d_maos < 0.08 * sw and d_pulsos >= 0.3 * sw:
            el = np.hypot((lh[t, 0, 0] - pose[t, 15, 0]) * W, (lh[t, 0, 1] - pose[t, 15, 1]) * H)
            er = np.hypot((rh[t, 0, 0] - pose[t, 16, 0]) * W, (rh[t, 0, 1] - pose[t, 16, 1]) * H)
            if el <= er:
                rh[t] = np.nan; src[t, 1] = 0
            else:
                lh[t] = np.nan; src[t, 0] = 0
            n += 1
    return n


def video_para_tensor(caminho, n_frames=T, lado=S):
    """Lê um vídeo (mp4, wmv...) e devolve o MESMO formato dos tensores do MALTA no Hugging Face:
    32 frames amostrados uniformemente (como UniformTemporalSubsample) e redimensionados para 224 x 224 RGB."""
    cap = cv2.VideoCapture(str(caminho))
    quadros = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        quadros.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (lado, lado), interpolation=cv2.INTER_AREA))
    cap.release()
    if not quadros:
        raise ValueError(f"não consegui ler frames de {caminho}")
    idx = np.floor(np.linspace(0, len(quadros) - 1, n_frames)).astype(int)
    return np.stack([quadros[i] for i in idx])


def processar(caminho, aspecto, escala, usar_plano_b, complexidade):
    """.pt do MALTA (ou .mp4/.wmv, convertido para o mesmo formato) -> keypoints."""
    bruto = ler_tensor(caminho) if str(caminho).lower().endswith(".pt") else video_para_tensor(caminho)
    return processar_array(bruto, aspecto, escala, usar_plano_b, complexidade)


def processar_array(video224, aspecto, escala, usar_plano_b, complexidade):
    """video224: (32, 224, 224, 3) RGB uint8, no formato dos tensores do MALTA."""
    video = preparar(cortar_faixas_pretas(video224), aspecto, escala)
    H, W = video.shape[1:3]
    pose = np.full((T, 33, 4), np.nan, np.float32)
    lh = np.full((T, 21, 3), np.nan, np.float32)
    rh = np.full((T, 21, 3), np.nan, np.float32)
    face = np.full((T, 468, 3), np.nan, np.float32)
    src = np.zeros((T, 2), np.int8)
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=complexidade,
                                        smooth_landmarks=True, refine_face_landmarks=False,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol, \
         mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1,
                                  min_detection_confidence=0.3) as maos:
        for t, f in enumerate(video):
            f = np.ascontiguousarray(f)
            r = hol.process(f)
            pose[t] = lm_array(r.pose_landmarks, 33, 4)
            lh[t] = lm_array(r.left_hand_landmarks, 21, 3)
            rh[t] = lm_array(r.right_hand_landmarks, 21, 3)
            face[t] = lm_array(r.face_landmarks, 468, 3)
            src[t, 0] = 1 if r.left_hand_landmarks else 0
            src[t, 1] = 1 if r.right_hand_landmarks else 0
            if usar_plano_b and r.pose_landmarks is not None:
                for k, (lado, arr) in enumerate((("L", lh), ("R", rh))):
                    if src[t, k] == 0:
                        b = plano_b_mao(f, pose[t], lado, maos)
                        if b is not None:
                            arr[t] = b
                            src[t, k] = 2
    corrigidos = corrigir_maos_duplicadas(pose, lh, rh, src, W, H)
    return dict(pose=pose, lh=lh, rh=rh, face=face, hand_src=src, frame_wh=np.array([W, H], np.int32),
                maos_corrigidas=np.int32(corrigidos))


def linha_registro(d, chave, npz_rel, r, segundos):
    p, lh, rh, src = d["pose"], d["lh"], d["rh"], d["hand_src"]
    cl, nl = cobertura_ativa(p, lh, POSE_L_WRIST)
    cr, nr = cobertura_ativa(p, rh, POSE_R_WRIST)
    return dict(arquivo=chave, npz=npz_rel, sinal=r.sinal, label=r.label, dictionary=r.dictionary, familia=r.familia,
                pose_pct=float((~np.isnan(p[:, 0, 0])).mean()), face_pct=float((~np.isnan(d["face"][:, 0, 0])).mean()),
                lh_pct=float((src[:, 0] > 0).mean()), rh_pct=float((src[:, 1] > 0).mean()),
                lh_ativo_frames=nl, lh_ativo_cob=cl, rh_ativo_frames=nr, rh_ativo_cob=cr,
                plano_b_frames=int((src == 2).sum()), maos_corrigidas=int(d.get("maos_corrigidas", 0)),
                frame_w=int(d["frame_wh"][0]), frame_h=int(d["frame_wh"][1]), segundos=segundos, erro="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=r"D:\ProjetoMestrado\data\malta_35")
    ap.add_argument("--saida", default=None, help="padrão: <raiz>\\kp")
    ap.add_argument("--aspecto", type=float, default=16 / 9, help="proporção original assumida (largura/altura)")
    ap.add_argument("--escala", type=int, default=2)
    ap.add_argument("--complexidade", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--sem_plano_b", action="store_true")
    ap.add_argument("--limite", type=int, default=None, help="N vídeos por dicionário (teste)")
    ap.add_argument("--refazer", action="store_true")
    ap.add_argument("--corrigir_existentes", action="store_true",
                    help="só aplica corrigir_maos_duplicadas aos .npz já gerados e refaz o registry (sem MediaPipe)")
    a = ap.parse_args()

    raiz = Path(a.raiz)
    saida = Path(a.saida) if a.saida else raiz / "kp"
    saida.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(raiz / "manifest.csv", encoding="utf-8-sig")
    man = man[man.status.isin(["baixado", "ja_existia"])].copy()
    if not a.corrigir_existentes:
        man["pt"] = [raiz / Path(str(x).replace("\\", "/")) for x in man.local_path]
        faltando = ~man.pt.map(Path.exists)
        if faltando.any():
            print(f"Aviso: {faltando.sum()} .pt do manifest não estão em {raiz} e serão pulados")
            man = man[~faltando]
    if a.limite:
        man = man.sample(frac=1, random_state=0).groupby("dictionary").head(a.limite)

    reg_path = saida / "registry.csv"
    registros = pd.read_csv(reg_path, encoding="utf-8-sig").to_dict("records") if reg_path.exists() and not a.refazer else []
    feitos = {r["arquivo"] for r in registros}
    t_ini = time.time()
    for i, r in enumerate(man.itertuples(), 1):
        rel = Path(str(r.local_path).replace("\\", "/"))
        destino = saida / rel.parent.name / (rel.stem + ".npz")
        chave = str(rel.as_posix())
        npz_rel = str(destino.relative_to(saida).as_posix())
        if a.corrigir_existentes:
            if not destino.exists():
                continue
            d = dict(np.load(destino, allow_pickle=False))
            W, H = d["frame_wh"]
            d["maos_corrigidas"] = np.int32(corrigir_maos_duplicadas(d["pose"], d["lh"], d["rh"], d["hand_src"], W, H))
            np.savez_compressed(destino, **d)
            antigo = next((x for x in registros if x["arquivo"] == chave), {})
            reg = linha_registro(d, chave, npz_rel, r, antigo.get("segundos", np.nan))
            registros = [x for x in registros if x["arquivo"] != chave] + [reg]
            continue
        if destino.exists() and chave in feitos and not a.refazer:
            continue
        t0 = time.time()
        try:
            d = processar(raiz / rel, a.aspecto, a.escala, not a.sem_plano_b, a.complexidade)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(man)}] ERRO {chave}: {e}")
            registros = [x for x in registros if x["arquivo"] != chave] + [dict(arquivo=chave, sinal=r.sinal, erro=str(e))]
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destino, **d, sinal=r.sinal, arquivo=chave, dictionary=r.dictionary, familia=r.familia)
        reg = linha_registro(d, chave, npz_rel, r, round(time.time() - t0, 2))
        registros = [x for x in registros if x["arquivo"] != chave] + [reg]
        if i % 10 == 0 or i == len(man):
            pd.DataFrame(registros).to_csv(reg_path, index=False, encoding="utf-8-sig")
        print(f"[{i}/{len(man)}] {r.sinal:10s} {r.dictionary:16s} mãoE={reg['lh_pct']:.2f} mãoD={reg['rh_pct']:.2f} "
              f"planoB={reg['plano_b_frames']:2d}  {reg['segundos']:.1f}s")
    pd.DataFrame(registros).to_csv(reg_path, index=False, encoding="utf-8-sig")
    print(f"Concluído em {time.time() - t_ini:.0f}s -> {saida}")


if __name__ == "__main__":
    main()
