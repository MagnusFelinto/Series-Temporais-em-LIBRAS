"""
minds_baixar.py — Caminho A, passo 1: baixa os dados para repetir o experimento do MALTA-Lab
(treinar no MINDS-Libras e testar nos vídeos do MALTA-LIBRAS com as mesmas 20 palavras).

O que baixa (tudo para <raiz>\\brutos\\):
  1. malta\\   os 145 vídeos do MALTA com as 20 palavras do MINDS, exatamente a lista do repositório
               ISLR_LIBRAS (dataset_intersections\\MINDS_matched_labels.csv):
                 - 118 tensores .pt do Hugging Face (MALTA-Lab/MALTA_LIBRAS, pasta malta_libras_complete)
                 - 27 vídeos .mp4 do V-LIBRASIL (UFPE), pelos links de video_downloads\\links_videos_vlibrasil.csv
               O teste do MALTA-Lab usa 131 deles (sem UFV e USP); a coluna no_teste_deles marca quais.
  2. minds_zip\\ o MINDS-Libras completo do Zenodo (registro 2667329, 12 arquivos .zip, um por sinalizante,
               cerca de 65 GB). O download continua de onde parou se for interrompido.

Saída: <raiz>\\manifest_malta.csv e <raiz>\\log_download.txt

Uso:
    python minds_baixar.py --raiz D:\\ProjetoMestrado\\data\\minds_malta --islr D:\\ProjetoMestrado\\references\\ISLR_LIBRAS
    python minds_baixar.py ... --so_malta        (só o conjunto de teste, ~0,7 GB)
    python minds_baixar.py ... --so_minds        (só o MINDS)
"""
import argparse
import re
import time
import unicodedata
from pathlib import Path, PurePosixPath

import pandas as pd
import requests

MINDS_PALAVRAS = ["acontecer", "aluno", "amarelo", "america", "aproveitar", "bala", "banco", "banheiro", "barulho",
                  "cinco", "conhecer", "espelho", "esquina", "filho", "maca", "medo", "ruim", "sapo", "vacina",
                  "vontade"]
ZENODO_REGISTRO = "2667329"
HF_REPO, HF_PREFIXO = "MALTA-Lab/MALTA_LIBRAS", "malta_libras_complete/"
EXT_VIDEO = {".mp4", ".wmv", ".mkv", ".avi", ".mov", ".mpg", ".mpeg", ".webm", ".1", ".2", ".pt"}
FAMILIA = {"Acessibilidades2": "INES", "Acessibilidades3": "INES", "UFSC": "UFSC", "UFSC_V2": "UFSC",
           "SpreadTheSign": "SpreadTheSign", "UFV": "UFV", "USP": "USP", "UFPE": "UFPE"}
FORA_DO_TESTE_DELES = {"UFV", "USP"}


def norm(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def palavra_minds(texto):
    """Qual das 20 palavras do MINDS aparece no texto (a mais longa, se houver mais de uma)."""
    t = re.sub(r"[^a-z]", "", norm(texto))
    achadas = [p for p in MINDS_PALAVRAS if p in t]
    return max(achadas, key=len) if achadas else None


def stem_limpo(nome):
    n = unicodedata.normalize("NFC", str(nome)).strip().lower()
    while True:
        suf = PurePosixPath(n).suffix
        if suf and suf in EXT_VIDEO:
            n = n[: -len(suf)]
        else:
            return n


def primeiro_caminho(x):
    x = str(x).strip()
    if x.startswith("[") and x.endswith("]"):
        x = x[1:-1].split("', '")[0].split('", "')[0]
    return x.strip().strip("'\"")


def baixar_url(url, destino, log, tentativas=3):
    """Download com retomada (HTTP Range) e conferência de tamanho."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    for t in range(tentativas):
        try:
            ja = parcial.stat().st_size if parcial.exists() else 0
            cab = {"Range": f"bytes={ja}-"} if ja else {}
            with requests.get(url, stream=True, timeout=60, headers=cab) as r:
                if r.status_code == 416:            # já estava completo
                    parcial.rename(destino)
                    return True
                r.raise_for_status()
                modo = "ab" if (ja and r.status_code == 206) else "wb"
                total = int(r.headers.get("Content-Length", 0)) + (ja if modo == "ab" else 0)
                with open(parcial, modo) as f:
                    for bloco in r.iter_content(1 << 20):
                        f.write(bloco)
            if total and parcial.stat().st_size < total:
                raise IOError(f"incompleto: {parcial.stat().st_size} de {total} bytes")
            parcial.replace(destino)
            return True
        except Exception as e:  # noqa: BLE001
            log(f"  erro ({t + 1}/{tentativas}) {url}: {e}")
            time.sleep(5 * (t + 1))
    return False


# ----------------------------------------------------------------------------- 1. MALTA (teste)
HF_PREFIXO_SUBSET = "malta_libras_MINDS_subset/"


def baixar_malta(raiz, islr, log):
    """Baixa os 145 vídeos do MALTA com as 20 palavras do MINDS.

    Ordem de busca de cada vídeo:
      1. pasta malta_libras_MINDS_subset/ do Hugging Face, pelo nome do tensor usado pelo MALTA-Lab
         (coluna tensor_path de MINDS_matched_labels_with_tensors.csv): é o conjunto de teste deles;
      2. pasta malta_libras_complete/ do Hugging Face, pela fonte e pelo nome do vídeo original;
      3. UFPE: o vídeo original no site do V-LIBRASIL.
    """
    from huggingface_hub import HfApi, hf_hub_download

    lista = pd.read_csv(islr / "dataset_intersections" / "MINDS_matched_labels_with_tensors.csv")
    lista["orig"] = lista.path.map(primeiro_caminho)
    lista["tensor"] = lista.tensor_path.map(lambda p: PurePosixPath(primeiro_caminho(p)).name)
    lista["sinal"] = lista["label"].map(palavra_minds).fillna(lista.orig.map(lambda p: palavra_minds(Path(p).name)))
    if lista.sinal.isna().any():
        log(f"Aviso: {lista.sinal.isna().sum()} linhas sem palavra reconhecida; serão ignoradas")
        lista = lista[lista.sinal.notna()]
    log(f"MALTA: {len(lista)} vídeos de {lista.sinal.nunique()} palavras; "
        f"{(~lista.dictionary.isin(FORA_DO_TESTE_DELES)).sum()} entram no teste do MALTA-Lab")
    destino_dir = raiz / "brutos" / "malta"

    log("Listando arquivos do Hugging Face...")
    todos = HfApi().list_repo_files(HF_REPO, repo_type="dataset")
    subset = {PurePosixPath(f).name.lower(): f for f in todos if f.startswith(HF_PREFIXO_SUBSET) and f.endswith(".pt")}
    completo = {}
    for f in todos:
        if f.startswith(HF_PREFIXO) and f.endswith(".pt"):
            partes = f[len(HF_PREFIXO):].split("/")
            completo.setdefault((partes[0].lower(), stem_limpo(partes[-1])), f)
    log(f"  {len(subset)} tensores em {HF_PREFIXO_SUBSET} | {len(completo)} em {HF_PREFIXO}")

    def baixar_hf(hf_path, local):
        tmp = hf_hub_download(HF_REPO, hf_path, repo_type="dataset", local_dir=raiz / "brutos" / "_hf_tmp")
        local.parent.mkdir(parents=True, exist_ok=True)
        Path(tmp).replace(local)

    links = None
    linhas = []
    for r in lista.itertuples():
        partes = PurePosixPath(r.orig).parts
        stem = stem_limpo(PurePosixPath(r.orig).name)
        base = f"{r.dictionary}__{re.sub(r'[^a-z0-9]+', '_', norm(stem))}"
        existentes = [p for p in (destino_dir / f"{base}.pt", destino_dir / f"{base}.mp4") if p.exists()]
        if r.dictionary == "UFPE":                                   # nome antigo dos vídeos da UFPE
            m = re.match(r"(.*)_(\d+)\.\w+$", PurePosixPath(r.orig).name)
            chave = re.sub(r"[^a-z0-9]+", "_", norm(m.group(1))).strip("_") if m else ""
            k = int(m.group(2)) - 1 if m else 0
            antigo = destino_dir / f"UFPE__{chave}_{k + 1}.mp4"
            if antigo.exists():
                existentes.append(antigo)
        local, status, origem_dl = None, "nao_encontrado", ""
        if existentes:
            local, status = existentes[0], "ja_existia"
        else:
            tentativas = []
            if r.tensor.lower() in subset:
                tentativas.append(("hf_subset", subset[r.tensor.lower()]))
            pasta = partes[partes.index("datasets") + 1].lower() if "datasets" in partes else ""
            if (pasta, stem) in completo:
                tentativas.append(("hf_completo", completo[(pasta, stem)]))
            for origem_dl, hf_path in tentativas:
                local = destino_dir / f"{base}.pt"
                try:
                    baixar_hf(hf_path, local)
                    status = "baixado"
                    break
                except Exception as e:  # noqa: BLE001
                    status = f"erro: {e}"
            if status != "baixado" and r.dictionary == "UFPE":
                if links is None:
                    links = pd.read_csv(islr / "video_downloads" / "links_videos_vlibrasil.csv")
                    links["chave"] = links["sign"].map(lambda s: re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_"))
                cand = links[links.chave == chave].link.tolist()
                if k < len(cand):
                    local, origem_dl = destino_dir / f"{base}.mp4", "vlibrasil"
                    status = "baixado" if baixar_url(cand[k], local, log) else "erro_vlibrasil"
        log(f"  {r.dictionary:16s} {r.sinal:10s} {stem:30s} {status} {origem_dl}")
        linhas.append(dict(arquivo=str(local.relative_to(raiz)) if local is not None else "", sinal=r.sinal,
                           dictionary=r.dictionary, familia=FAMILIA.get(r.dictionary, r.dictionary), origem=r.orig,
                           tensor_deles=r.tensor, status=status, fonte_download=origem_dl,
                           no_teste_deles=r.dictionary not in FORA_DO_TESTE_DELES))

    man = pd.DataFrame(linhas)
    man.to_csv(raiz / "manifest_malta.csv", index=False, encoding="utf-8-sig")
    ok = man.status.isin(["baixado", "ja_existia"])
    log(f"MALTA pronto: {ok.sum()} de {len(man)} vídeos; no teste do MALTA-Lab: "
        f"{(ok & man.no_teste_deles).sum()} de {man.no_teste_deles.sum()}")
    if (~ok).any():
        log("Faltando:\n" + man[~ok][["dictionary", "sinal", "origem", "status"]].to_string(index=False))


# ----------------------------------------------------------------------------- 2. MINDS (treino)
def baixar_minds(raiz, log):
    info = requests.get(f"https://zenodo.org/api/records/{ZENODO_REGISTRO}", timeout=60).json()
    arquivos = info.get("files", [])
    log(f"MINDS (Zenodo {ZENODO_REGISTRO}): {len(arquivos)} arquivos, "
        f"{sum(f.get('size', 0) for f in arquivos) / 1e9:.1f} GB")
    pasta = raiz / "brutos" / "minds_zip"
    for f in sorted(arquivos, key=lambda x: x["key"]):
        nome, tam = f["key"], f.get("size", 0)
        url = f["links"].get("content") or f["links"].get("self")
        destino = pasta / nome
        if destino.exists() and (not tam or destino.stat().st_size == tam):
            log(f"  {nome}: já baixado")
            continue
        log(f"  {nome}: baixando {tam / 1e9:.2f} GB...")
        t0 = time.time()
        ok = baixar_url(url, destino, log)
        log(f"  {nome}: {'ok' if ok else 'FALHOU'} em {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=r"D:\ProjetoMestrado\data\minds_malta")
    ap.add_argument("--islr", default=r"D:\ProjetoMestrado\references\ISLR_LIBRAS", help="pasta do repositório ISLR_LIBRAS")
    ap.add_argument("--so_malta", action="store_true")
    ap.add_argument("--so_minds", action="store_true")
    a = ap.parse_args()
    raiz, islr = Path(a.raiz), Path(a.islr)
    raiz.mkdir(parents=True, exist_ok=True)
    arq_log = open(raiz / "log_download.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        arq_log.write(msg + "\n")
        arq_log.flush()

    log(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if not a.so_minds:
        baixar_malta(raiz, islr, log)
    if not a.so_malta:
        baixar_minds(raiz, log)


if __name__ == "__main__":
    main()
