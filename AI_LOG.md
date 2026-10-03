# AI_LOG — como usamos IA neste assignment

Usamos o Claude (Claude Code, no terminal) como par de programação ao longo do trabalho. O que segue é um resumo honesto de onde a IA entrou e de como o trabalho foi conduzido.

# O caminho: sintético primeiro, real depois

No começo tivemos dificuldade em lidar com os dados reais, o MOT17 traz sete vídeos em condições muito diferentes (câmera parada e câmera móvel, densidades que vão de poucos pedestres a cenas lotadas, oclusões longas), cada vídeo em três versões de detector, e não estava claro como transformar o `gt.txt`/`det.txt` e os milhares de quadros em algo que um modelo de movimento recorrente pudesse treinar e ser avaliado de forma justa. Em vez de travar aí fizemos todas as partes primeiro com o vídeo sintético da Parte 0. Isso deixou o pipeline inteiro pronto (gerador de trajetórias com oclusão, detector simulado, métricas de IDF1 e ID switches, rastreador ingênuo por IoU, `MovimentoRNN`, treino com BPTT truncado, ablação RNN/GRU/LSTM, galeria de falhas, teste de estresse) sem depender dos dados reais.

Depois, com o pipeline funcionando, adaptamos parte por parte para o MOT17. O ganho de ter feito o sintético antes foi grande: como já sabíamos que o matching por IoU, o IDF1 e o laço do rastreador estavam certos, as diferenças de resultado eram atribuíveis aos dados, não a bug. Os principais episódios em que recorremos à IA:

## Como preparamos e usamos os prompts

### 1. Carregar os dados reais (MOT17)

- "Tenho o pipeline rodando no vídeo sintético: o `gt` é uma lista de dicts `{frame, id, x, y, w, h, vis}` e as detecções são `{frame, x, y, w, h, conf}`. Quero carregar as sequências reais do MOT17 na mesma interface, sem mudar os nomes de campo. Escreva o parser do `gt.txt` (mantendo só pedestres, classe 1, com flag ativa) e do `det.txt` (com limiar de score). Valide antes de me entregar: confira que os ranges de frame batem, que os ids são contíguos por sequência, e me mostre 3 linhas de exemplo lado a lado (sintético vs real) para eu conferir o alinhamento. Se algum campo não encaixar na interface, me aponte em vez de adaptar sozinho."

A interface-alvo e os filtros (classe de pedestre, flag, limiar) são decisões nossas, a IA faz o trabalho mecânico de parsing e prova que o alinhamento bate antes de a gente confiar.

### 2. Split sem vazamento de vídeo

- "O MOT17 tem 7 vídeos únicos × 3 detectores, as 3 versões de um vídeo são o mesmo vídeo. Quero um split que não vaze o mesmo vídeo entre treino e validação e que use só uma versão de detector (SDP) para não contar o mesmo quadro três vezes. Reserve para validação uma sequência de câmera parada (MOT17-09) e uma de câmera móvel (MOT17-10), que o modelo temporal nunca vê no treino. Implemente isso e me mostre as contagens (quantas trajetórias e quantos quadros sobram por conjunto). Se a minha escolha de VAL deixar algum regime sub-representado no treino, me avise, não troque por mim."

O eixo "câmera parada × câmera móvel" e as sequências de validação foram escolha nossa, o prompt só pede a implementação e uma checagem de que a escolha não desbalanceia o treino.

### 3. Adaptar as Partes (rastreador e métricas) aos dados reais

- "Essas funções já passaram nos testes do sintético (`rastreador_rnn`, IDF1, contagem de switches). Quero rodá-las nas sequências reais sem mudar a lógica de associação, só trocando a fonte dos dados. Me mostre o diff mínimo por parte e marque explicitamente qualquer linha onde o comportamento (não só o dado) mudaria. Se não der para adaptar sem mexer na lógica, pare e me explique por quê."

### 4. Limpeza e organização

- "Quero uma passada de simplificação: um único laço de `treinar` que sirva para todas as configurações da ablação (RNN, GRU, LSTM, cada janela T), com `hidden` ajustado para igualar o número de parâmetros, e um `rastreador_rnn` que receba qualquer célula no lugar da LSTM. Nomes de variáveis claros, sem comentário óbvio, uma célula de markdown por item do enunciado. Faça a refatoração mas garanta equivalência: rode a versão antiga e a nova com a mesma seed e me mostre que a perda por passo bate. Se alguma mudança mexer em célula que já rodou, não execute, me diga quais eu preciso reexecutar e em que ordem."

### 5. Decisões de projeto (apenas discutidas com a IA, mas decididas por nós)

Aqui o formato muda: pedíamos o trade-off, não a escolha.

- "Na Parte 4 vou corrigir um diagnóstico da galeria de falhas. Temos duas opções: só aumentar `max_age` para sobreviver a oclusões longas ou fazer associação em cascata, casando primeiro as tracks vistas no quadro anterior e só depois as que estão sem observação. Na Parte 2, aumentar `max_age` sozinho piorou o MOT17-10. Para cada opção, me liste o que melhora, o que piora e qual métrica no meu VAL revelaria o efeito colateral. Não recomende uma, deixe que eu decido depois de ver os números."

- "Vou escolher a célula e a janela para o resto do PA pela consistência entre seeds e janelas (IDF1 estável em VAL), não pelo melhor número isolado. Me dê o argumento contra esse critério, quando ele me faria escolher mal."

As decisões que saíram desse formato e foram nossas: usar a densidade da sequência como eixo de dificuldade, adotar LSTM com T=16 como modelo do resto do PA, por ser consistente entre janelas e seeds, corrigir a Parte 4 com cascata + `max_age=30` (em vez de só `max_age` maior), cobrindo a oclusão mediana e o p75, selecionar a época/configuração pela estabilidade do IDF1 em validação e, no teste de estresse, dividir `max_age` pela taxa de subamostragem para manter o mesmo tempo de vídeo. A leitura de que "o gradiente que some é real, mas não é o gargalo desta tarefa, o que quebra a RNN simples é o free-running fora da janela de treino" também foi conclusão nossa a partir dos números.

### 6. Leitura de resultados 

- "Aqui estão as tabelas da ablação (erro às cegas em 1/10/30 quadros, IDF1, switches, norma do gradiente por seed). Não interprete. Só me aponte: onde há não-monotonicidade que eu deveria justificar e qualquer número que provavelmente seja ruído de uma sequência única e não efeito real."

Isso é o oposto de "explique meus resultados": mantém a autoria da análise com a gente e usa a IA como checagem de sanidade. Foi assim que marcamos, por exemplo, que o vale do MOT17-09 em 1/2 era provavelmente ruído de uma sequência só.

### 7. Revisão crítica antes de manter

- "Revise esta célula procurando só bugs silenciosos: vazamento do mesmo vídeo entre treino e validação, estado da RNN não resetado entre tracks, `max_age` em unidade errada depois da subamostragem, detecção contada em versão de detector duplicada. Para cada achado: célula, linha, por que quebra. Não refatore nem 'melhore' estilo, apenas me indique onde e o que foi encontrado."

## O que a IA não fez

As escolhas de trilha, de eixos de ablação, de modalidade retirada e de correção foram nossas; os resultados foram rodados e lidos por nós; e revisamos cada célula gerada antes de manter. Todo código do repositório é reproduzível a partir dele e entendemos o que cada função faz — o notebook está organizado justamente para que cada decisão tenha a justificativa ao lado do código.
