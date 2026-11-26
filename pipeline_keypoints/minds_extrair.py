"""
minds_extrair.py — Caminho A, passo 2: extrai os keypoints do MINDS (treino) e dos 145 vídeos do MALTA (teste)
com EXATAMENTE o mesmo pipeline da Etapa 1 (extrair_keypoints.py).

Todo vídeo passa pelo mesmo formato dos tensores do MALTA no Hugging Face antes do MediaPipe:
32 frames amostrados uniformemente, 224 x 224, depois desachatado para 16:9 e ampliado 2x (796 x 448).
Assim treino (MINDS) e teste (MALTA) são tratados de forma idêntica, como no experimento do MALTA-Lab.

Entrada : <raiz>\\brutos\\minds_zip\\*.zip  e  <raiz>\\brutos\\malta\\*  +  <raiz>\\manifest_malta.csv
Saída   : <raiz>\\kp\\minds\\<palavra>\\*.npz, <raiz>\\kp\\malta\\<palavra>\\*.npz
          <raiz>\\kp\\registry.csv   (uma linha por vídeo; familia = MINDS_S01..S12 ou a fonte do MALTA)
          <raiz>\\classes.csv        (as 20 palavras do MINDS)
Pode ser interrompido e rodado de novo (pula o que já foi feito).

Uso:  python minds_extrair.py --raiz D:\\ProjetoMestrado\\data\\minds_malta
"""
import argparse
import re
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import extrair_keypoints as ek
from minds_baixar import MINDS_PALAVRAS, norm, palavra_minds

EXT_VIDEO = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".webm")
CLASSES = {p: i for i, p in enumerate(sorted(MINDS_PALAVRAS))}


def abrir_zip(zpath):
    """Abre um zip do MINDS corrigindo os deslocamentos errados dos arquivos.

    Os zips do MINDS com mais de 4 GB foram criados no macOS com deslocamentos de 32 bits estourados:
    o zipfile do Python lê um endereço errado e falha com "Truncated file header", "Bad magic number"
    ou "Overlapped entries". Para cada arquivo, procuramos o cabeçalho local verdadeiro (assinatura PK\\3\\4
    e o mesmo nome) entre os candidatos deslocamento mod 2^32 + k * 2^32.
    """
    z = zipfile.ZipFile(zpath)
    tamanho = Path(zpath).stat().st_size
    corrigidos = 0
    with open(zpath, "rb") as f:
        for info in z.infolist():
            nome = info.filename.encode("utf-8" if info.flag_bits & 0x800 else "cp437")
            for cand in sorted({(info.header_offset % 2**32) + k * 2**32 for k in range(4)}):
                if cand + 30 > tamanho:
                    continue
                f.seek(cand)
                cab = f.read(30)
                if cab[:4] != b"PK\x03\x04":
                    continue
                n_nome = int.from_bytes(cab[26:28], "little")
                if f.read(n_nome) == nome:
                    if cand != info.header_offset:
                        info.header_offset = cand
                        corrigidos += 1
                    break
    if corrigidos:
        # o Python 3.12 confere sobreposição com o fim calculado de cada arquivo (_end_offset);
        # recalculamos esse fim com os deslocamentos corrigidos
        ordem = sorted(z.filelist, key=lambda i: i.header_offset)
        for atual, prox in zip(ordem, ordem[1:] + [None]):
            if hasattr(atual, "_end_offset"):
                atual._end_offset = prox.header_offset if prox is not None else z.start_dir
        print(f"{Path(zpath).name}: {corrigidos} deslocamentos corrigidos")
    return z


def sinalizador(texto):
    m = re.search(r"sinalizador\D{0,3}0*(\d{1,2})", norm(texto))
    return int(m.group(1)) if m else None


def salvar(d, destino, chave, sinal, dictionary, familia):
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino, **d, sinal=sinal, arquivo=chave, dictionary=dictionary, familia=familia)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=r"D:\ProjetoMestrado\data\minds_malta")
    ap.add_argument("--sem_minds", action="store_true")
    ap.add_argument("--sem_malta", action="store_true")
    ap.add_argument("--limite_minds", type=int, default=None, help="no máximo N vídeos por zip (teste rápido)")
    a = ap.parse_args()
    raiz = Path(a.raiz)
    kp = raiz / "kp"
    kp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(sinal=list(CLASSES), label=list(CLASSES.values()))).to_csv(
        raiz / "classes.csv", index=False, encoding="utf-8-sig")
    reg_path = kp / "registry.csv"
    registros = pd.read_csv(reg_path, encoding="utf-8-sig").to_dict("records") if reg_path.exists() else []
    feitos = {r["arquivo"] for r in registros if str(r.get("erro", "")) in ("", "nan")}
    t_ini, n_novos = time.time(), 0

    def registrar(d, chave, npz_rel, sinal, dictionary, familia, conjunto, no_teste_deles, segundos):
        nonlocal registros, n_novos
        r = SimpleNamespace(sinal=sinal, label=CLASSES[sinal], dictionary=dictionary, familia=familia)
        linha = ek.linha_registro(d, chave, npz_rel, r, segundos)
        linha.update(conjunto=conjunto, no_teste_deles=no_teste_deles)
        registros = [x for x in registros if x["arquivo"] != chave] + [linha]
        n_novos += 1
        if n_novos % 20 == 0:
            pd.DataFrame(registros).to_csv(reg_path, index=False, encoding="utf-8-sig")

    def erro(chave, sinal, e):
        nonlocal registros
        print(f"  ERRO {chave}: {e}")
        registros = [x for x in registros if x["arquivo"] != chave] + [dict(arquivo=chave, sinal=sinal, erro=str(e))]

    # ---------------------------------------------------------------- MALTA (teste)
    if not a.sem_malta and (raiz / "manifest_malta.csv").exists():
        man = pd.read_csv(raiz / "manifest_malta.csv", encoding="utf-8-sig")
        man = man[man.status.isin(["baixado", "ja_existia"])]
        print(f"MALTA: {len(man)} vídeos")
        for i, r in enumerate(man.itertuples(), 1):
            chave = r.arquivo.replace("\\", "/")
            npz_rel = f"malta/{r.sinal}/{Path(chave).stem}.npz"
            if chave in feitos and (kp / npz_rel).exists():
                continue
            t0 = time.time()
            try:
                d = ek.processar(raiz / chave, 16 / 9, 2, True, 1)
            except Exception as e:  # noqa: BLE001
                erro(chave, r.sinal, e)
                continue
            salvar(d, kp / npz_rel, chave, r.sinal, r.dictionary, r.familia)
            registrar(d, chave, npz_rel, r.sinal, r.dictionary, r.familia, "malta", bool(r.no_teste_deles),
                      round(time.time() - t0, 2))
            print(f"[MALTA {i}/{len(man)}] {r.dictionary:16s} {r.sinal:10s} {time.time() - t0:.1f}s", flush=True)

    # ---------------------------------------------------------------- MINDS (treino)
    zips = sorted((raiz / "brutos" / "minds_zip").glob("*.zip")) if not a.sem_minds else []
    tmp = raiz / "_tmp_video"
    tmp.mkdir(exist_ok=True)
    for zpath in zips:
        try:
            z = abrir_zip(zpath)
        except zipfile.BadZipFile:
            print(f"{zpath.name}: zip incompleto ou corrompido; rode o minds_baixar.py de novo")
            continue
        membros = [m for m in z.namelist() if m.lower().endswith(EXT_VIDEO) and not m.startswith("__MACOSX")]
        if a.limite_minds:
            membros = membros[: a.limite_minds]
        print(f"{zpath.name}: {len(membros)} vídeos")
        for j, m in enumerate(membros, 1):
            sinal = palavra_minds(Path(m).name) or palavra_minds(m)
            num = sinalizador(m) or sinalizador(zpath.name)
            chave = f"minds/{zpath.name}/{m}"
            if sinal is None or num is None:
                erro(chave, sinal, f"não reconheci palavra ({sinal}) ou sinalizador ({num}) no nome")
                continue
            familia = f"MINDS_S{num:02d}"
            npz_rel = f"minds/{sinal}/{familia}__{re.sub(r'[^a-z0-9]+', '_', norm(Path(m).stem))}.npz"
            if chave in feitos and (kp / npz_rel).exists():
                continue
            t0 = time.time()
            arq_tmp = tmp / Path(m).name
            try:
                with z.open(m) as src, open(arq_tmp, "wb") as dst:
                    while True:
                        bloco = src.read(1 << 20)
                        if not bloco:
                            break
                        dst.write(bloco)
                d = ek.processar(arq_tmp, 16 / 9, 2, True, 1)
            except Exception as e:  # noqa: BLE001
                erro(chave, sinal, e)
                continue
            finally:
                arq_tmp.unlink(missing_ok=True)
            salvar(d, kp / npz_rel, chave, sinal, "MINDS", familia)
            registrar(d, chave, npz_rel, sinal, "MINDS", familia, "minds", False, round(time.time() - t0, 2))
            print(f"[{zpath.name} {j}/{len(membros)}] {familia} {sinal:10s} {time.time() - t0:.1f}s", flush=True)

    reg = pd.DataFrame(registros)
    reg.to_csv(reg_path, index=False, encoding="utf-8-sig")
    ok = reg[reg.erro.fillna("") == ""] if "erro" in reg else reg
    print(f"\nConcluído em {(time.time() - t_ini) / 60:.1f} min. Vídeos com keypoints: "
          f"MINDS={int((ok.conjunto == 'minds').sum())} ({ok[ok.conjunto == 'minds'].familia.nunique()} sinalizadores), "
          f"MALTA={int((ok.conjunto == 'malta').sum())}; erros={len(reg) - len(ok)}")
    if len(ok):
        print(ok.groupby("conjunto")[["pose_pct", "lh_pct", "rh_pct"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
