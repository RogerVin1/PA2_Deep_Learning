# AI_LOG

## Revisão da Parte 2 (modelo de movimento + rastreador)

Depois de plugar a LSTM no rastreador, o ganho sobre o ingênuo era pequeno (IDF1 0,52 vs 0,48 no MOT17-09) e não sabíamos se era limitação do método ou erro nosso. Pedimos para a IA revisar a Parte 2 procurando bugs.

Ela comparou o modelo treinado com dois baselines simples e mostrou que a LSTM era **pior do que só copiar a última caixa**. Os problemas que ela apontou:

1. **A rede previa a caixa absoluta.** Entre dois quadros a caixa anda ~0,001 em coordenada normalizada, então esse erro quase não gerava gradiente na Smooth-L1 e a rede aprendia algo próximo da identidade. Correção: a rede recebe e prevê o *deslocamento*, dividido pelo desvio-padrão dos deslocamentos do treino.
2. **O rastreador alimentava a última caixa duas vezes na LSTM**: uma no update e outra na hora de prever. No treino cada caixa entra uma vez só. Correção: cada track guarda a previsão feita no mesmo passo em que a caixa entrou.

Com as duas correções, o modelo passou a bater "copiar a última caixa" e "velocidade constante". O IDF1 subiu para 0,545 (MOT17-09) e 0,489 (MOT17-10). Conferimos os números rodando de novo e mantivemos a explicação nos comentários do código.
