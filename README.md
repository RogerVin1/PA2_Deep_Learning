# PA2 — Identidade ao longo do tempo: detecção, recorrência e rastreamento

## **Disciplina:** Aprendizado Profundo

## **Dupla:** Nicholas Costa e Roger Augusto

Trilha escolhida (Parte 2): **A — RNN como modelo de movimento**. Um estado recorrente por track prevê a caixa do próximo quadro, a associação usa IoU entre a caixa prevista e as detecções observadas.

## Estrutura

| Arquivo | O que é |
|---|---|
| `Assignment2_DL.ipynb` | Notebook principal, Partes 0 a 5. Toda tabela, curva e figura da apresentação. |
| `metrics.py` | Implementação própria de IDF1, ID switches e fragmentações (e também IoU, average_precision, erro de contagem). |
| `inferencia.ipynb` | Recebe o caminho de uma sequência, devolve o vídeo com identidades coloridas de forma consistente e a contagem de objetos únicos. Não retreina. |
| `checkpoints/movimento_lstm.pt` | Pesos do modelo temporal final. |
| `checkpoints/ablacao/` | Checkpoints da grade da Parte 3 (3 células × 4 janelas × 3 seeds), reaproveitados pela Parte 4. |
| `AI_LOG.md` | Como a IA foi usada. |

## Ambiente

Python ≥ 3.10. Dependências: PyTorch, torchvision, NumPy, SciPy, Matplotlib.

## Dados

MOT17 / MOTChallenge (https://motchallenge.net/data/MOT17/), sem conta.

- **Anotações** (~10 MB): suficientes para métricas, rastreamento e treino do
  modelo de movimento. Baixadas automaticamente pelo notebook.
- **Pacote completo** (~5,5 GB): necessário apenas para o detector torchvision e
  para o vídeo de `inferencia.ipynb`.

Usamos como fonte de detecção padrão o detector **SDP**, por ter o maior
recall/qualidade entre os três públicos (DPM, FRCNN, SDP), isso deixa a entrada
do rastreador mais limpa, de modo que os erros de rastreamento são atribuíveis à
associação, não ao detector.

**Split por sequência**, pois quadros vizinhos são quase idênticos,
então separar por quadro vazaria informação. Usamos apenas as versões `-SDP`
(as 3 versões de cada vídeo no MOT17 são o mesmo vídeo com detectores diferentes).
A validação são **MOT17-09 (câmera parada) e MOT17-10 (câmera móvel)**, que o
modelo temporal nunca vê no treino — cobrindo os dois regimes de câmera.

## Métrica

Implementadas à mão em `metrics.py`:
- **IDF1**: atribuição global 1-para-1 entre identidades previstas e verdadeiras
  ao longo da sequência (Hungarian via `linear_sum_assignment`).
- **ID switches**: trocas de identidade, por quadro, via matching por IoU.
- **Fragmentações**: interrupções de uma track verdadeira (casada → perdida →
  casada), independentes de troca de id.
- Validadas nos três casos sintéticos da Parte 0 (predição = gabarito;
  2 identidades trocadas; 1 track partida), com os resultados esperados
  (IDF1 = 1.00 / 0.60 / 0.80; switches = 0 / 2 / 1).

Também incluímos `average_precision` (mAP de detecção, estilo VOC) e o erro de
contagem de identidades únicas.

## O notebook

| Parte | Conteúdo |
|---|---|
| 0 | Ambiente sintético: gerador de elipses com oclusão por profundidade, simulador de detector, métricas + 3 testes à mão, baseline no piso fácil, sweep de velocidade. |
| 1 | Baseline por quadro: detecções públicas (SDP) e torchvision, rastreador ingênuo (IoU + Hungarian), IDF1/switches/fragmentações, erro de contagem, gráfico do descolamento mAP × IDF1. |
| 2 | Memória temporal (Trilha A): modelo de movimento recorrente (`MovimentoRNN`, previsão residual do deslocamento), rastreador com caixa prevista, comparação lado a lado com a baseline nas mesmas sequências. |
| 3 | Ablação **Eixo 1: RNN vs GRU vs LSTM** com o mesmo orçamento de parâmetros, variando a janela de BPTT truncado T ∈ {4, 8, 16, 32}, 3 seeds (média ± desvio). Mede erro de previsão sem observação (oclusão simulada), IDF1/switches no rastreamento, e a norma do gradiente ‖∂L_T/∂h_{T−k}‖ em função de k. Conclusão: a RNN simples quebra na janela **curta** (não na longa) por sair da distribuição de treino sob free-running, e o "gradiente que some" explica só parte do fenômeno: a dependência útil cabe em poucos passos. |
| 4 | Galeria de falhas e horizonte de memória. Horizonte medido de duas formas: **analítica** (curva do gradiente vs k) e **empírica** (sobrevivência da identidade por duração da lacuna vs distribuição de oclusões do dataset). Galeria com três tipos de falha diagnosticados por medida (track morreu / nasceu track nova / track vizinha roubou). **Correção implementada**: associação em cascata (matching cascade, Wojke et al. 2017, reescrita) + `max_age=30`, com antes/depois, e IDF1 do MOT17-10 sobe de 0,489 para 0,524. |
| 5 | Teste de estresse: **queda de taxa de quadros** (vídeo subamostrado a 1/2 e 1/5), sem retreinar. Curva de degradação do IDF1 para ingênuo, LSTM e LSTM+cascata, e erro de previsão do próximo passo vs "copiar última" e "velocidade constante", separando culpa do modelo de movimento da culpa da associação. Discute por que um modelo aprendido em Δt fixo quebra quando Δt muda e se alimentar Δt na recorrência resolveria. |
