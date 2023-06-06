#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

struct Movie {
    int id;
    int start_time;
    int end_time;
    int category;
};

void print_selected_movies(const vector<Movie>& movies, vector<int>& categories);

void select_movies(vector<Movie>& movies, vector<int>& categories, int index, int num_selected, vector<int>& best_solution);

int main() {
    int n, m;
    cin >> n >> m;

    vector<Movie> movies(n);
    for (int i = 0; i < n; i++) {
        cin >> movies[i].start_time >> movies[i].end_time >> movies[i].category;
        movies[i].id = i + 1;
    }

    vector<int> categories(m);
    for (int i = 0; i < m; i++) {
        cin >> categories[i];
    }

    vector<int> best_solution(m);
    vector<int> current_solution(m);

    #pragma omp parallel
    {
        #pragma omp single nowait
        select_movies(movies, categories, 0, 0, best_solution);
    }

    print_selected_movies(movies, best_solution);

    return 0;
}

void print_selected_movies(const vector<Movie>& movies, vector<int>& categories) {
    int num_selected = 0;
    vector<int> selected_movies;

    #pragma omp parallel for reduction(+:num_selected)
    for (int i = 0; i < movies.size(); i++) {
        const Movie& movie = movies[i];
        int category = movie.category;

        #pragma omp critical
        if (categories[category - 1] > 0) {
            num_selected++;
            selected_movies.push_back(i);
            categories[category - 1]--;
        }
    }

    cout << num_selected << endl;

    #pragma omp parallel for
    for (int i = 0; i < selected_movies.size(); i++) {
        const Movie& movie = movies[selected_movies[i]];
        cout << movie.id << " " << movie.category << endl;
    }
}


void select_movies(vector<Movie>& movies, vector<int>& categories, int index, int num_selected, vector<int>& best_solution) {
    if (index == movies.size()) {
        if (num_selected > 0) {
            // Verifica se a solução atual é melhor do que a melhor solução encontrada até agora
            int current_sum = 0;
            int best_sum = 0;
            for (int i = 0; i < categories.size(); i++) {
                current_sum += categories[i];
                best_sum += best_solution[i];
            }
            if (current_sum > best_sum) {
                #pragma omp critical
                best_solution = categories;
            }
        }
        return;
    }

    // Não selecionar o filme atual
    select_movies(movies, categories, index + 1, num_selected, best_solution);

    // Selecionar o filme atual, se possível
    if (num_selected < movies[index].category) {
        categories[movies[index].category - 1]++;
        select_movies(movies, categories, index + 1, num_selected + 1, best_solution);
        categories[movies[index].category - 1]--;
    }
}
