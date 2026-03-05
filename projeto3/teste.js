import tf from "@tensorflow/tfjs-node"
import { readFile } from "node:fs/promises"

function normalizarCliente(clientes, cliente) {
    const norm = (val, arr) => {
        const max = Math.max(...arr)
        const min = Math.min(...arr)
        return (val - min) / (max - min)
    }
    return [
        norm(cliente.idade, clientes.map(c => c.idade)),
        norm(cliente.renda_mensal, clientes.map(c => c.renda_mensal)),
        norm(cliente.score_credito, clientes.map(c => c.score_credito)),
        norm(cliente.divida_atual, clientes.map(c => c.divida_atual)),
        norm(cliente.meses_empregado, clientes.map(c => c.meses_empregado)),
        cliente.historico_inadimplencia
    ]
}

// Casos de teste baseados nas regras de negocio (regras.md)
const casos = [
    // Regra 1: score >= 650 E divida/renda < 0.35 → APROVADO
    {
        descricao: "Regra 1 - Score alto com baixa divida",
        cliente: { score_credito: 700, renda_mensal: 5000, divida_atual: 500, meses_empregado: 24, historico_inadimplencia: 0, idade: 35 },
        esperado: 1
    },

    // Regra 2: score >= 800 → APROVADO (mesmo com divida)
    {
        descricao: "Regra 2 - Score muito alto garante aprovacao",
        cliente: { score_credito: 850, renda_mensal: 4000, divida_atual: 1200, meses_empregado: 12, historico_inadimplencia: 0, idade: 40 },
        esperado: 1
    },

    // Regra 3: score < 500 E inadimplencia = 1 → REPROVADO
    {
        descricao: "Regra 3 - Score baixo com inadimplencia",
        cliente: { score_credito: 420, renda_mensal: 3000, divida_atual: 400, meses_empregado: 18, historico_inadimplencia: 1, idade: 28 },
        esperado: 0
    },

    // Regra 4: score < 350 → REPROVADO
    {
        descricao: "Regra 4 - Score muito baixo",
        cliente: { score_credito: 300, renda_mensal: 6000, divida_atual: 200, meses_empregado: 48, historico_inadimplencia: 0, idade: 42 },
        esperado: 0
    },

    // Regra 5: meses_empregado < 10 E score < 600 → REPROVADO
    {
        descricao: "Regra 5 - Emprego curto com score medio",
        cliente: { score_credito: 550, renda_mensal: 3500, divida_atual: 300, meses_empregado: 5, historico_inadimplencia: 0, idade: 25 },
        esperado: 0
    },

    // Regra 6: renda < 2000 E divida > 1000 → REPROVADO
    {
        descricao: "Regra 6 - Renda baixa com divida alta",
        cliente: { score_credito: 580, renda_mensal: 1500, divida_atual: 1500, meses_empregado: 20, historico_inadimplencia: 0, idade: 30 },
        esperado: 0
    },

    // Caso limite: score 800 com inadimplencia (regra 2 sobrepoe regra 3)
    {
        descricao: "Limite - Score 800 com inadimplencia (regra 2 deve prevalecer)",
        cliente: { score_credito: 800, renda_mensal: 5000, divida_atual: 500, meses_empregado: 24, historico_inadimplencia: 1, idade: 38 },
        esperado: 1
    },

    // Caso negativo claro: tudo ruim
    {
        descricao: "Negativo claro - Perfil de alto risco",
        cliente: { score_credito: 280, renda_mensal: 1200, divida_atual: 2000, meses_empregado: 2, historico_inadimplencia: 1, idade: 20 },
        esperado: 0
    },

    // Caso positivo claro: tudo bom
    {
        descricao: "Positivo claro - Perfil ideal",
        cliente: { score_credito: 950, renda_mensal: 12000, divida_atual: 200, meses_empregado: 120, historico_inadimplencia: 0, idade: 45 },
        esperado: 1
    }
]

async function executarTestes() {
    const clientes = JSON.parse(await readFile('./data/clientes.json'))
    const model = await tf.loadLayersModel('file://./modelo/model.json')

    let acertos = 0

    console.log('='.repeat(60))
    console.log('RESULTADO DOS TESTES')
    console.log('='.repeat(60))

    for (const caso of casos) {
        const normalizado = normalizarCliente(clientes, caso.cliente)
        const tensor = tf.tensor2d([normalizado])
        const previsao = model.predict(tensor)
        const probabilidade = (await previsao.data())[0]
        const resultado = probabilidade >= 0.5 ? 1 : 0
        const passou = resultado === caso.esperado

        if (passou) acertos++

        const status = passou ? 'PASSOU' : 'FALHOU'
        const esperadoStr = caso.esperado === 1 ? 'APROVADO' : 'REPROVADO'
        const resultadoStr = resultado === 1 ? 'APROVADO' : 'REPROVADO'

        console.log(`\n[${status}] ${caso.descricao}`)
        console.log(`  Esperado: ${esperadoStr} | Obtido: ${resultadoStr} (${(probabilidade * 100).toFixed(1)}%)`)
    }

    console.log('\n' + '='.repeat(60))
    console.log(`TOTAL: ${acertos}/${casos.length} testes passaram`)
    console.log('='.repeat(60))
}

await executarTestes()
