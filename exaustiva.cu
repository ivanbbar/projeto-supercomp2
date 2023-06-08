#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/copy.h>

using namespace std;

struct Movie
{
  int id;
  int start;
  int end;
  int category;
};

void fillTimeSlots(int &availableTimeSlots, int start, int end)
{
  for (int i = start; i < end; i++)
  {
    availableTimeSlots |= (1 << i);
  }
}

class ExhaustiveSearchGPU
{
private:
  int movieCount;
  int categoryCount;
  int *categoryAvailability;
  int *movieSchedules;
  int *movieCategories;
  int *maxCount;

public:
  ExhaustiveSearchGPU(int movieCount_, int categoryCount_, int *categoryAvailability_, int *movieSchedules_, int *movieCategories_, int *maxCount_)
      : movieCount(movieCount_),
        categoryCount(categoryCount_),
        categoryAvailability(categoryAvailability_),
        movieSchedules(movieSchedules_),
        movieCategories(movieCategories_),
        maxCount(maxCount_) {}

  __device__ void operator()(const int &config)
  {
    int availableTimeSlots = 0;
    int viewedCategories[16];

    for (int i = 0; i < categoryCount; i++)
    {
      viewedCategories[i] = categoryAvailability[i];
    }

    int selectedMoviesCount = 0;
    for (int i = 0; i < movieCount; i++)
    {
      if (config & (1 << i))
      {
        if (viewedCategories[movieCategories[i] - 1] > 0)
        {
          int currentSchedule = availableTimeSlots & movieSchedules[i];
          if (currentSchedule != 0)
            return;

          viewedCategories[movieCategories[i] - 1]--;
          availableTimeSlots |= movieSchedules[i];
          selectedMoviesCount++;
        }
      }
    }

    atomicMax(maxCount, selectedMoviesCount);
  }
};

void inputCategoryAvailability(thrust::host_vector<int> &categoryAvailability)
{
  for (int i = 0; i < categoryAvailability.size(); i++)
  {
    cin >> categoryAvailability[i];
  }
}

void inputMovies(vector<Movie> &moviesVector, int movieCount)
{
  for (int i = 0; i < movieCount; i++)
  {
    int start, end, category;
    cin >> start >> end >> category;

    if (start > end)
    {
      if (end == 0)
      {
        end = 24;
      }
      else
      {
        continue;
      }
    }

    Movie movie;
    movie.id = i + 1;
    movie.start = start;
    movie.end = end;
    movie.category = category;

    moviesVector.push_back(movie);
  }
}

void populateMoviesAndSchedules(thrust::host_vector<int> &movieSchedulesCPU, thrust::host_vector<int> &movieCategories, const vector<Movie> &moviesVector)
{
  for (int i = 0; i < moviesVector.size(); i++)
  {
    movieSchedulesCPU[i] = 0;
    fillTimeSlots(movieSchedulesCPU[i], moviesVector[i].start, moviesVector[i].end);
    movieCategories[i] = moviesVector[i].category;
  }
}

void printSelectedMovies(const vector<Movie> &moviesVector, const thrust::host_vector<int> &configVectorCPU, int maxCount)
{
  int maxConfig = -1;
  for (int i = 0; i < configVectorCPU.size(); i++)
  {
    if (configVectorCPU[i] == maxCount)
    {
      maxConfig = i;
      break;
    }
  }

  for (int i = 0; i < moviesVector.size(); i++)
  {
    if (maxConfig & (1 << i))
    {
      cout << moviesVector[i].id << " " << moviesVector[i].category << endl;
    }
  }
}

int main()
{
  int movieCount, categoryCount;
  cin >> movieCount >> categoryCount;

  vector<Movie> moviesVector;
  thrust::host_vector<int> categoryAvailability(categoryCount);
  inputCategoryAvailability(categoryAvailability);

  inputMovies(moviesVector, movieCount);
  int actualMovieCount = moviesVector.size();

  thrust::host_vector<int> movieCategories(actualMovieCount);
  thrust::host_vector<int> movieSchedulesCPU(actualMovieCount);
  populateMoviesAndSchedules(movieSchedulesCPU, movieCategories, moviesVector);

  thrust::device_vector<int> gpuPossibilitiesVector(pow(2, actualMovieCount));
  thrust::sequence(gpuPossibilitiesVector.begin(), gpuPossibilitiesVector.end());

  thrust::device_vector<int> categoryAvailabilityGPU(categoryAvailability);
  thrust::device_vector<int> movieSchedulesGPU(movieSchedulesCPU);
  thrust::device_vector<int> movieCategoriesGPU(movieCategories);
  thrust::device_vector<int> gpuMaxCount(1, 0);

  ExhaustiveSearchGPU searchFunctor(actualMovieCount, categoryCount, raw_pointer_cast(categoryAvailabilityGPU.data()),
                                    raw_pointer_cast(movieSchedulesGPU.data()),
                                    raw_pointer_cast(movieCategoriesGPU.data()), raw_pointer_cast(gpuMaxCount.data()));

  thrust::for_each(gpuPossibilitiesVector.begin(), gpuPossibilitiesVector.end(), searchFunctor);

  thrust::host_vector<int> finalConfigVectorCPU = gpuPossibilitiesVector;
  int maxCount = *thrust::max_element(finalConfigVectorCPU.begin(), finalConfigVectorCPU.end());

  cout << maxCount << endl;
  printSelectedMovies(moviesVector, finalConfigVectorCPU, maxCount);

  return 0;
}
