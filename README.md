# botPython

## Como aplicar a mudança da navegação linear

A navegação linear já está implementada no `bot.py`. Para ajustar ao seu cliente/resolução, altere estes parâmetros no topo do arquivo:

- `LINEAR_CLICK_SCALE`: distância do clique à frente do personagem (menor = mais seguro, maior = mais rápido).
- `LINEAR_CLICK_JITTER`: tamanho do quadrado aleatório em volta do ponto principal.
- `LINEAR_CLOSE_JITTER`: jitter usado quando está perto do alvo para reduzir zigue-zague.
- `RESCUE_CLICK_SCALE`: distância usada nos cliques de emergência (quando OCR falha/trava).

### Sugestões práticas

- Se ainda travar em obstáculo: reduza `LINEAR_CLICK_SCALE` para `0.55` e `LINEAR_CLICK_JITTER` para `26`.
- Se ficar lento demais: aumente `LINEAR_CLICK_SCALE` para `0.68`.
- Se oscilar perto do checkpoint: reduza `LINEAR_CLOSE_JITTER` para `14`.

## Validar rapidamente

```bash
python -m py_compile bot.py
```

Ao iniciar, o bot agora imprime o perfil de navegação ativo no log (`[NAV] ...`) para você confirmar os valores carregados.
