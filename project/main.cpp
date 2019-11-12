#include <iostream>
#include "NeuralNetwork.hpp"
using namespace std;

TestData* GetTestData() {
  TestData* data = new TestData;
  for (int i = 0; i < 3; i++) {
    data->input.push_back(rand() % 2);
  }
  data->rightAnswere =
      ((data->input[0] && data->input[1]) || data->input[2]) ? 1 : 0;
  return data;
}

int main() {
  NeuralNetwork n(3, vector<int>{4, 4, 2});
  vector<float> input = {1, 0, 0};
  // n.Calculate(input);
  // n.PrintNetworkStatus();
  for (int i = 0; i < TRAIN_CICLES; i++) n.Train(GetTestData(), false);
  // n.Train(input, 0, true);
  n.PrintNetworkWithParams();
  n.Evaluate(input, 0);
}
