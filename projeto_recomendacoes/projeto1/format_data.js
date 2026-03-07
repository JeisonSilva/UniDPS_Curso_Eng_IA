
export function normalizePessoa(pessoa) {
    const idadeMin = 25
    const idadeMax = 40

    const idadeNormalizada =
        Number(((pessoa.idade - idadeMin) / (idadeMax - idadeMin)).toFixed(2))
    const corAzul = pessoa.cor === "azul" ? 1 : 0
    const corVermelho = pessoa.cor === "vermelho" ? 1 : 0
    const corVerde = pessoa.cor === "verde" ? 1 : 0
    const localizacaoSaoPaulo = pessoa.localizacao === "Sao Paulo" ? 1 : 0
    const localizacaoRio = pessoa.localizacao === "Rio de Janeiro" ? 1 : 0

    return [[
        idadeNormalizada,
        corAzul,
        corVermelho,
        corVerde,
        localizacaoSaoPaulo,
        localizacaoRio
    ]]
}

export function encodeCategoria(categoria) {
    const categorias = {
        premium: [1, 0, 0],
        medium: [0, 1, 0],
        basic: [0, 0, 1]
    }

    return categorias[categoria]
}

export function exibirReferenciasTreinamento(pessoasReais, tensores, categorias, labels) {
    const referencias = pessoasReais.map((pessoa, index) => {
        const categoriaIndex = categorias[index].findIndex((valor) => valor === 1)

        return {
            idade: pessoa.idade,
            cor: pessoa.cor,
            localizacao: pessoa.localizacao,
            dadosNormalizados: JSON.stringify(tensores[index]),
            categoria: labels[categoriaIndex]
        }
    })

    console.log("Referencias de treinamento:")
    console.table(referencias)
}

export function exibirNovaPessoa(dadosReais, dadosNormalizados) {
    console.log("Nova pessoa:")
    console.table([{
        idade: dadosReais.idade,
        cor: dadosReais.cor,
        localizacao: dadosReais.localizacao,
        dadosNormalizados: JSON.stringify(dadosNormalizados[0])
    }])
}

export function exibirResultadoFormatado(resultados) {
    console.log("Resultado da previsao:")
    console.table(
        resultados.map((resultado, index) => ({
            posicao: index + 1,
            classificacao: resultado.label,
            probabilidade: `${(resultado.prob * 100).toFixed(2)}%`
        }))
    )
}


export const labelsNomes = ["premium", "medium", "basic"]


