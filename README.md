# botPython

## Como aplicar a mudança da navegação linear

A navegação linear já está implementada no `bot.py`. Para ajustar ao seu cliente/resolução, altere estes parâmetros no topo do arquivo:

- `LINEAR_CLICK_SCALE`: distância do clique à frente do personagem (menor = mais seguro, maior = mais rápido).
- `LINEAR_CLICK_JITTER`: tamanho do quadrado aleatório em volta do ponto principal.
- `LINEAR_CLOSE_JITTER`: jitter usado quando está perto do alvo para reduzir zigue-zague.
- `RESCUE_CLICK_SCALE`: distância usada nos cliques de emergência (quando OCR falha/trava).
- `AXIS_LOCK_CROSS_JITTER`: jitter mínimo no eixo transversal quando o bot está em linha reta (ex.: andar só no X mantendo Y).
- `MIN_CLICK_COMPONENT`: tamanho mínimo do clique por eixo para evitar cliques curtos demais que não movem o personagem.

### Sugestões práticas

- Se ainda travar em obstáculo: reduza `LINEAR_CLICK_SCALE` para `0.65` e `LINEAR_CLICK_JITTER` para `24`.
- Se ficar lento demais: aumente `LINEAR_CLICK_SCALE` para `0.80`.
- Se oscilar perto do checkpoint: reduza `LINEAR_CLOSE_JITTER` para `14`.

## Validar rapidamente

```bash
python -m py_compile bot.py
```

Ao iniciar, o bot agora imprime o perfil de navegação ativo no log (`[NAV] ...`) para você confirmar os valores carregados.


## Rota mais curta (novo comportamento)

Agora o bot tenta primeiro uma rota direta (reta) até o destino final.
Se houver bloqueio/obstáculo, ele cai automaticamente para a rota por checkpoints.

Esse fluxo fica nos logs como:
- `[ROUTE][SHORT] tentando rota direta ...`
- `[ROUTE][SHORT] rota direta falhou -> fallback para checkpoints.`


### Movimento em linha reta por eixo (novo)

- Quando já está alinhado no mesmo **Y** do destino, o bot força direção apenas em **X** (`E`/`W`) para fazer trajetória reta.
- Quando já está alinhado no mesmo **X** do destino, o bot força direção apenas em **Y** (`N`/`S`).
- O parâmetro `AXIS_LOCK_CROSS_JITTER` mantém uma pequena variação no eixo cruzado para não travar clique perfeito, mas quase em linha reta.


### Anti-travamento (novo ajuste)

- Quando detecta `STUCK`, o bot agora faz **rescue direcional para o alvo** (não aleatório).
- Se a distância piorar em sequência, ele aumenta temporariamente força/distância do clique.
