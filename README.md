# Maratona de Filmes

Você quer passar um final de semana assistindo ao máximo de filmes possível, mas há restrições quanto aos horários disponíveis e ao número de títulos que podem ser vistos em cada categoria (comédia, drama, ação, etc).

Entrada: Um inteiro N representando o número de filmes disponíveis para assistir e N trios de inteiros (H[i], F[i], C[i]), representando a hora de início, a hora de fim e a categoria do i-ésimo filme. Além disso, um inteiro M representando o número de categorias e uma lista de M inteiros representando o número máximo de filmes que podem ser assistidos em cada categoria.

Saída: Um inteiro representando o número máximo de filmes que podem ser assistidos de acordo com as restrições de horários e número máximo por categoria.

## Heurística Gulosa

A primeira implementação da heurística para nosso projeto consiste em uma implementação gulosa (Greedy).

Implemente uma versão gulosa que ordena os filmes por hora de fim crescente e escolhe aqueles que começam primeiro e não conflitam com os filmes já escolhidos, além de verificar se há vagas disponíveis na categoria do filme.

## Aleatoriedade

Como vimos em aula, aleatoriedade é uma estratégia bastante comum para construção de algoritmos de busca local, podendo ser usada de forma isolada ou de forma complementar a outra estratégia de varredura de um espaço de soluções.

Essa implementação consiste na adaptação da heurística gulosa de nosso projeto. A proposta é que você modifique a sua heurística gulosa de modo que ao longo da seleção de um filme você tenha 25% de chance de pegar outro filme qualquer que respeite o horário. Isso fará com que sua heurística tenha um pouco mais de exploration e possamos ter alguns resultados melhores.

Importante: é essencial que você guarde todos os inputs usados ao longo do projeto, para que possa comparar o desempenho de seus algoritmos conforme mudamos a heurística. Ou seja, todas as heurísticas devem ser submetidas aos mesmos arquivos de input. O seu resultado deve ser comparado sob duas perspectivas, no mínimo: (i) tempo de execução em função do aumento de filmes e de categorias e (ii) tempo de tela (isto é, será que estamos conseguindo ocupar bem as 24h do dia assitindo filmes?).
