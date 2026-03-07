# Cenario: Aprovacao de Credito Pessoal

Regras de negocio usadas para rotular os dados de treino. O modelo aprende a replicar essas regras a partir dos exemplos.

---

## Campos (Features)

| Campo | Descricao | Tipo | Normalizacao | Impacto |
|---|---|---|---|---|
| `score_credito` | Pontuacao de credito do cliente (0 a 1000) | Numerico | `score / 1000` | Maior score indica historico de pagamentos positivo. Principal indicador de aprovacao. |
| `renda_mensal` | Renda mensal bruta em reais | Numerico | `renda / renda_maxima_do_dataset` | Renda mais alta aumenta capacidade de pagamento. |
| `divida_atual` | Total de dividas ativas em reais | Numerico | `divida / renda_mensal` (razao divida/renda) | Divida alta em relacao a renda reduz chance de aprovacao. |
| `meses_empregado` | Tempo de emprego continuo em meses | Numerico | `meses / 240` (maximo considerado: 20 anos) | Emprego estavel por mais tempo reduz risco de inadimplencia. |
| `historico_inadimplencia` | 1 se ja teve inadimplencia registrada, 0 se nunca teve | Binario | Usar o valor diretamente (0 ou 1) | Inadimplencia anterior e forte indicador de rejeicao. |
| `idade` | Idade do cliente em anos | Numerico | `idade / 100` | Fator secundario. Idades entre 30-55 tendem a ter perfis mais estaveis. |

---

## Regras de Aprovacao

| # | Descricao | Condicao | Resultado |
|---|---|---|---|
| 1 | Score alto com baixa divida | `score_credito >= 650` E `divida_atual / renda_mensal < 0.35` | **Aprovado** |
| 2 | Score muito alto garante aprovacao mesmo com alguma divida | `score_credito >= 800` | **Aprovado** |
| 3 | Score baixo com inadimplencia rejeita | `score_credito < 500` E `historico_inadimplencia = 1` | **Reprovado** |
| 4 | Score muito baixo rejeita independente de outros fatores | `score_credito < 350` | **Reprovado** |
| 5 | Emprego curto com score medio rejeita | `meses_empregado < 10` E `score_credito < 600` | **Reprovado** |
| 6 | Renda muito baixa com divida rejeita | `renda_mensal < 2000` E `divida_atual > 1000` | **Reprovado** |

---

## Features para o Modelo

1. Score de credito normalizado
2. Renda mensal normalizada
3. Razao divida/renda
4. Meses empregado normalizados
5. Historico de inadimplencia (0 ou 1)
6. Idade normalizada

---

## Label

- **Campo:** `aprovado`
- **Tipo de problema:** Classificacao binaria

| Valor | Significado |
|---|---|
| `1` | Credito aprovado |
| `0` | Credito reprovado |

---

## Dicas de Estudo

- Observe que o campo `score_credito` e o mais correlacionado com o resultado. Tente treinar o modelo **sem ele** e veja como a acuracia cai.
- Experimente adicionar mais exemplos de casos "aprovado" com score medio (600-700) para ver se o modelo aprende a fronteira de decisao.
- A razao divida/renda e uma feature derivada. Criar features derivadas e uma tecnica chamada **feature engineering**.
- Tente mudar a arquitetura da rede (mais ou menos neuronios) e observe o impacto na loss final.
- Com apenas 30 exemplos, o modelo pode memorizar os dados (**overfitting**). Adicione mais clientes no JSON para testar isso.
