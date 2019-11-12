#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK
#include "NeuralLayer.hpp"

typedef struct testData_t {
  vector<float> input;
  int rightAnswere;
} TestData;

class NeuralNetwork {
 private:
  InputNeuralLayer _inputLayer;
  vector<NeuralLayer> _layers;

 public:
  NeuralNetwork(int inputNum, vector<int> layersDim)
      : _inputLayer(inputNum, 0) {
    int id = 1;
    _layers.push_back(NeuralLayer(layersDim[0], inputNum, id++));
    for (int ind = 1; ind < layersDim.size(); ind++) {
      _layers.push_back(NeuralLayer(layersDim[ind], layersDim[ind - 1], id++));
    }
  }
  void Calculate(vector<float> input) {
    _inputLayer.SetValues(input);
    _layers[0].CalculateLayer(_inputLayer.GetNodes());
    for (int ind = 1; ind < _layers.size(); ind++) {
      _layers[ind].CalculateLayer(_layers[ind - 1].GetNodes());
    }
  }

  void Train(TestData* data, bool verbose) {
    Train(data->input, data->rightAnswere, verbose);
  }
  void Train(vector<float> inputs, int rightOutput, bool verbose) {
    Calculate(inputs);
    if (verbose) printf("----------------------------------------------\n");
    vector<float> desideredValues;
    for (int i = 0; i < _layers[_layers.size() - 1].GetNodeNum(); i++) {
      if (i == rightOutput)
        desideredValues.push_back(1);
      else
        desideredValues.push_back(0);
    }
    for (int i = _layers.size() - 1; i > 0; i--) {
      desideredValues = _layers[i].TrainLayer(_layers[i - 1].GetNodes(),
                                              desideredValues, verbose);
    }
    _layers[0].TrainLayer(_inputLayer.GetNodes(), desideredValues, verbose);
    if (verbose) printf("----------------------------------------------\n");
  }
  void PrintNetworkStatus() {
    printf("\n");
    _inputLayer.PrintLayer();
    printf("\n");
    for (NeuralLayer& l : _layers) {
      l.PrintLayer();
      printf("\n");
    }
  }
  void PrintNetworkWithParams() {
    printf("TRAIN_CICLES:\t\t\t%d\n", TRAIN_CICLES);
    printf("PARAM_INTENSITY_CORRECT:\t%.5f\n", PARAM_INTENSITY_CORRECT);
    printf("PARAM_INTENSITY_ERROR:\t\t%.5f\n", PARAM_INTENSITY_ERROR);
    _layers[_layers.size() - 1].PrintLayer();
  }
  float Evaluate(vector<float> inputs, int rightOutput) {
    Calculate(inputs);
    float eval = 0;
    int counter = 0;
    for (NeuralNode n : _layers[_layers.size() - 1].GetNodes()) {
      float error =
          (rightOutput != counter++) ? 1 - n.GetValue() : n.GetValue();
      eval += error / _layers[_layers.size() - 1].GetNodeNum();
    }
    cout << "[VALUATION OF NEURAL NETWORK: " << (eval * 100) << "%]" << endl;
    return eval;
  }
};

#endif  // NEURAL_NETWORK