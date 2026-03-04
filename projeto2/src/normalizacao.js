export function normalizarIdade(idade) {
    return idade / 100;
}

export function normalizarValor(valor, valorMaximo) {
    return valor / valorMaximo;
}

export function normalizarIndice(indice, total) {
    if (total <= 1) {
        return 0;
    }

    return indice / (total - 1);
}
