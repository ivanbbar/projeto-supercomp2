#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <bitset>

using namespace std;

struct Movie {
    int id;
    int start_time;
    int end_time;
    int category;
    std::bitset<24> schedule;
};

bool compare_movie(Movie a, Movie b) {
    return a.end_time < b.end_time;
}

std::bitset<24> generate_schedule(int start, int end) {
    std::bitset<24> schedule;

    if (start == end) {
        schedule.set(start);
        return schedule;
    }
    
    for (int i = start; i < end; i++) {
        schedule.set(i);
    }

    return schedule;
}

vector<Movie> read_movies(int num_movies);

void select_movies(vector<Movie>& movies, vector<int>& categories, vector<int>& num_movies, int& marathon_availability);

bool is_compatible_with_marathon(const std::bitset<24>& schedule, const int marathon_availability);

void add_movie_to_marathon(const Movie& movie, const int category, vector<int>& num_movies, int& marathon_availability, vector<int>& categories);

void print_selected_movies(const vector<Movie>& movies, const vector<int>& categories, const int num_selected_movies);

void greedy_heuristic(vector<Movie>& movies, vector<int>& categories, vector<int>& num_movies, int& marathon_availability);


void select_movies(vector<Movie>& movies, vector<int>& categories, vector<int>& num_movies, int& marathon_availability) {
    const int num_movies_total = movies.size();

    for (int i = 0; i < num_movies_total; i++) {
        const Movie& movie = movies[i];
        const int category = movie.category;
        const std::bitset<24> schedule = movie.schedule;

        if (num_movies[category - 1] == 0) {
            continue;
        }

        if (marathon_availability == 0 || is_compatible_with_marathon(schedule, marathon_availability)) {
            add_movie_to_marathon(movie, category, num_movies, marathon_availability, categories);
        }
    }
}

bool is_compatible_with_marathon(const std::bitset<24>& schedule, const int marathon_availability) {
    const std::bitset<24> temp_schedule(marathon_availability);
    const std::bitset<24> intersection = temp_schedule & schedule;
    return intersection.none();
}

void add_movie_to_marathon(const Movie& movie, const int category, vector<int>& num_movies, int& marathon_availability, vector<int>& categories) {
    marathon_availability |= movie.schedule.to_ulong();
    categories.push_back(category);
    num_movies[category - 1]--;
}

void print_selected_movies(const vector<Movie>& movies, const vector<int>& categories, const int num_selected_movies) {
    cout << num_selected_movies << endl;

    for (int i = 0; i < num_selected_movies; i++) {
        const Movie& movie = movies[i];
        cout << movie.id << " " << categories[i] << endl;
    }
}

void greedy_heuristic(vector<Movie>& movies, vector<int>& categories, vector<int>& num_movies, int& marathon_availability) {
    select_movies(movies, categories, num_movies, marathon_availability);
    const int num_selected_movies = categories.size();
    print_selected_movies(movies, categories, num_selected_movies);
}

vector<Movie> read_movies(int num_movies) {
    vector<Movie> movies;

    for (int i = 0; i < num_movies; i++) {
        int start_time, end_time, category;
        cin >> start_time >> end_time >> category;

        if (start_time > end_time) {
            if (end_time == 0) {
                end_time = 24;
            } else if (start_time == -1 || end_time == -1) {
                continue;
            } else {
                continue;
            }
        }

        Movie movie;
        movie.id = i + 1;
        movie.start_time = start_time;
        movie.end_time = end_time;
        movie.category = category;
        movie.schedule = generate_schedule(start_time, end_time);

        movies.push_back(movie);
    }

    sort(movies.begin(), movies.end(), compare_movie);

    return movies;
}

int main() {
    int n, m;
    cin >> n >> m;

    vector<int> num_movies(m);
    for (int i = 0; i < m; i++) {
        cin >> num_movies[i];
    }

    vector<Movie> movies = read_movies(n);

    vector<int> categories;
    int marathon_availability = 0;

    greedy_heuristic(movies, categories, num_movies, marathon_availability);

    return 0;
}

