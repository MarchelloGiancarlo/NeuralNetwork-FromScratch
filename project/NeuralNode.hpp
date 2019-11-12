#ifndef NEURAL_NODE
#define NEURAL_NODE
#include <bits/stdc++.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include "Configuration.hpp"
using namespace std;

namespace nodeId {
int NODE_ID_COUNTER = 1000;
}

class Node {
 private:
  float _value;

 protected:
  int _id;

 public:
  Node() {
    _value = 0;
    _id = nodeId::NODE_ID_COUNTER++;
  }
  void SetValue(float value) {
    if (value <= 1 && value >= 0)
      _value = value;
    else
      cout << "[ERROR] try to set value: " << value << endl;
    // cout << _id << "-SET:" << _value << endl;
  }
  float GetValue() { return _value; }
  void printNode() { printf("[NODE:%d]:%.5f\n", _id, _value); }
};

class NeuralNode : public Node {
 private:
  float* _neuralWeights;
  int _neuralInputNumber;
  float Normalize(float x) { return 1 / (1 + exp(-x)); }
  void ModifyWeight(float intensity, int index) {
    _neuralWeights[index] += intensity;
    if (_neuralWeights[index] > 1) _neuralWeights[index] = 1;
    if (_neuralWeights[index] < 0) _neuralWeights[index] = 0;
  }

 public:
  NeuralNode(int neuralInputs) : Node() {
    _neuralInputNumber = neuralInputs;
    _neuralWeights = new float[neuralInputs];
    for (int ind = 0; ind < _neuralInputNumber; ind++) {
      _neuralWeights[ind] = (float)rand() / (float)RAND_MAX;
    }
  }
  void CalculateNode(vector<NeuralNode> prevNodes) {
    float sum = 0;
    for (int ind = 0; ind < _neuralInputNumber; ind++) {
      sum += prevNodes[ind].GetValue() * _neuralWeights[ind];
    }
    SetValue(Normalize(sum));
  }
  vector<float>* TrainNode(vector<NeuralNode>& prevNodes, float desideredValues,
                           bool verbose) {
    vector<float>* desideredChangeOnPrevNodes = new vector<float>;
    float error = desideredValues - GetValue();
    if (verbose)
      printf("[train node:%d]-[desValue=%.5f\tvalue=%.5f\terror=%.5f]\n", _id,
             desideredValues, GetValue(), error);
    // if value = 0.8 & desValue = 0.3 -> error = -0.5
    for (int ind = 0; ind < _neuralInputNumber; ind++) {
      float prevNodeEffect = prevNodes[ind].GetValue() - GetValue();
      // if value = 0.8 & prevNodeValue = 0.5 -> prevNodeEffect = 0.3
      if (verbose)
        printf("\t[prevNode[%d]=%.5f\tweight=%.5f\teffect=%.5f]\n", ind,
               prevNodes[ind].GetValue(), _neuralWeights[ind], prevNodeEffect);
      float change = prevNodeEffect * _neuralWeights[ind] * abs(error);
      if ((error > 0 && prevNodeEffect >= 0) ||
          (error < 0 && prevNodeEffect <= 0)) {
        ModifyWeight(
            PARAM_INTENSITY_CORRECT * abs(error) / (1 - _neuralWeights[ind]),
            ind);
        // if (verbose) printf("<%.5f>\n", PARAM_INTENSITY_CORRECT *
        // abs(error));
        if (verbose)
          printf(
              "\t\t[CHANGE: increased weight to %.5f - "
              "prevNodeEffect=%.5f]\n",
              _neuralWeights[ind], change);

        desideredChangeOnPrevNodes->push_back(change);
      } else {
        ModifyWeight(-PARAM_INTENSITY_ERROR * abs(error) / _neuralWeights[ind],
                     ind);
        // if (verbose) printf("<%.5f>\n", -PARAM_INTENSITY_ERROR * abs(error));
        if (verbose)
          printf(
              "\t\t[CHANGE: decreased weight to %.5f - "
              "prevNodeEffect=%.5f]\n",
              _neuralWeights[ind], -change);
        desideredChangeOnPrevNodes->push_back(-change);
      }
    }
    return desideredChangeOnPrevNodes;
  }
  void printNode() {
    printf(" [NNODE:%d]:%.5f\n", _id, GetValue());
    for (int i = 0; i < _neuralInputNumber; i++) {
      printf("\tW[%d]: %f", i, _neuralWeights[i]);
      if (i + 1 % 5 == 0) printf("\n");
    }
    printf("\n");
  }
};

#endif  // NEURAL_NODE