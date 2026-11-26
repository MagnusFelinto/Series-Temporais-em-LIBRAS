# Reconhecimento de sinais isolados de Libras como classificação de séries temporais

Trabalho final da disciplina **Tópicos Especiais em ESSC II — IA Aplicada a Séries Temporais** (PPgCC UERN/UFERSA, 2026).

Cada vídeo do **MINDS-Libras** (20 palavras, 12 sinalizantes, 1.158 vídeos) vira uma **série temporal multivariada** de keypoints do MediaPipe Holistic (32 instantes × 321 atributos: corpo, mãos e formato das mãos). O objetivo é reconhecer a palavra **em pessoas que nunca apareceram no treino**.

## Principais resultados

F1 macro no teste (média ± desvio, 4 rodadas por sinalizante × 3 sementes):

| Modelo | F1 macro (%) |
|---|---|
| Classe majoritária (baseline ingênua) | 0,5 |
| 1-NN + DTW (baseline clássica) | 61,5 ± 6,0 |
| Árvore de Decisão / Random Forest / XGBoost (32 lags) | 71,2 / 89,0 / 87,6 |
| CNN-1D / WaveNet | 91,3 / 87,9 |
| SimpleRNN / LSTM / GRU | 46,4 / 78,2 / 84,3 |
| Conv1D + GRU (híbrido) | 89,0 |
| **CNN-1D + 1ª diferença (melhor resultado)** | **92,6 ± 3,6** |
| Bônus: Transformer de patches multicanal / PatchTST | 90,4 / 79,0 |

- **Pré-processamento:** média móvel e FFT não ajudam; a 1ª diferença melhora a CNN-1D (+2,2 pontos, p = 0,034).
- **Horizonte (classificação antecipada):** com 50% do sinal, a CNN-1D já tem ~90% de F1.

A tabela completa está em `resultados/08_tabela_geral.csv` e as figuras do artigo em `figuras/08_*.png`.

## Estrutura

```
notebooks/            00 a 08, na ordem de execução (cada um é autocontido)
dados/minds_st.npz    séries prontas (gerado pelo notebook 00)
resultados/           uma tabela .csv por notebook (e 08_*.tex para o artigo)
figuras/              figuras de cada notebook (08_* = versão do artigo)
pipeline_keypoints/   código que transforma os vídeos do MINDS nas séries (opcional)
```

| Notebook | Conteúdo | Treino |
|---|---|---|
| 00_preparacao_dados | séries do MINDS, divisão em 4 grupos de sinalizantes, `minds_st.npz` | — |
| 01_analise_exploratoria | caracterização temporal, estacionariedade (ADF/KPSS), ACF/PACF, espectro, atributos ergódicos, RF × XGBoost, agrupamento com DTW | CPU, minutos |
| 02_baselines_arvores | classe majoritária, 1-NN + DTW, Árvore de Decisão, RF, XGBoost | CPU, ~35 min |
| 03_redes_profundas | CNN-1D, WaveNet, SimpleRNN, LSTM, GRU, Conv1D + GRU, curvas de aprendizado | GPU, ~35 min |
| 04_ablacao | média móvel e FFT (com e sem), teste de Wilcoxon pareado | GPU, ~1h45 |
| 05_horizonte | classificação antecipada (12,5% a 100% do sinal), estratégias direta × modelo único | GPU, ~1h15 |
| 06_calibracao_diferencas | calibração de hiperparâmetros das redes + 1ª diferença | GPU, ~2h30 |
| 07_bonus_patchtst | PatchTST e Transformer de patches multicanal × CNN-1D (ganho × custo) | GPU, ~1h |
| 08_tabela_geral_figuras | tabela geral, resumo da ablação, figuras e tabelas LaTeX | — |

## Protocolo (resumo)

- **Divisão por sinalizante** (equivalente, para séries independentes, à divisão cronológica): 4 rodadas; em cada uma, 3 sinalizantes no teste, 2 na validação e 7 no treino. Nunca há sorteio aleatório.
- **Sem vazamento:** z-score ajustado só no treino de cada rodada; as técnicas de pré-processamento são aplicadas série a série.
- **Hiperparâmetros** escolhidos só na validação; *early stopping* na validação; o teste é usado uma vez.
- **Sementes fixas** (0, 1, 2). Os treinos em GPU não são bit a bit determinísticos: o mesmo modelo varia até ~0,8 ponto de F1 entre execuções.

## Como reproduzir

```
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_analise_exploratoria.ipynb   # e assim por diante
```

Os notebooks gravam cada treino assim que ele termina e **continuam de onde pararam** se forem interrompidos.

- **Google Colab:** copie a pasta para `Meu Drive/trabalho_series_temporais`. Se o Drive não montar, a primeira célula pede o envio do `minds_st.npz` e a última baixa os resultados em `.zip`.
- **GPU no Windows:** o TensorFlow não usa a GPU no Windows nativo. Use o WSL2 (Ubuntu) com `pip install "tensorflow[and-cuda]"` e aponte o `LD_LIBRARY_PATH` para as bibliotecas NVIDIA do ambiente:
  ```
  export LD_LIBRARY_PATH=$(find <venv>/lib/python3*/site-packages/nvidia -name lib -type d | paste -sd:):/usr/lib/wsl/lib
  ```

### Dos vídeos às séries (opcional)

O `minds_st.npz` já está no repositório. Para gerá-lo a partir dos vídeos originais (MINDS-Libras, Zenodo, registro 2667329, ~65 GB):

```
cd pipeline_keypoints
pip install -r requirements.txt
python minds_baixar.py --raiz <pasta> --so_minds         # baixa os 12 zips
python minds_extrair.py --raiz <pasta> --sem_malta       # MediaPipe Holistic: 32 frames por vídeo
python montar_dataset.py --raiz <pasta> --so_familia MINDS_ --k_suavizar 1 --saida ../dados/minds_bruto_k1.npz
```

Depois rode o notebook 00. Os zips do MINDS com mais de 4 GB têm deslocamentos internos de 32 bits estourados; o `minds_extrair.py` corrige isso ao abrir cada zip.

## Referências principais

- Rezende, T. M. et al. (2021). Development and validation of a Brazilian sign language database for human gesture recognition. *Neural Computing and Applications*.
- Bontempi, G.; Ben Taieb, S.; Le Borgne, Y.-A. (2013). Machine learning strategies for time series forecasting.
- Nielsen, A. (2019). *Practical Time Series Analysis*. O'Reilly.
- Ismail Fawaz, H. et al. (2019). Deep learning for time series classification: a review. *Data Mining and Knowledge Discovery*.
- Nie, Y. et al. (2023). A time series is worth 64 words: long-term forecasting with Transformers (PatchTST). *ICLR*.
- Kong, X. et al. (2025). Deep learning for time series forecasting: a survey.
