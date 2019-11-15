#ifndef NEURAL_NODE
#define NEURAL_NODE
#include <bits/stdc++.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#define sigmoid(x) (1 / (1 + exp(-x)))
#define sigmoidDer(x) (x * (1 - x))
using namespace std;

int __NODE_ID_IDENTITY = 1000;

class Node {
 private:
  float _value;

 protected:
  int _id;

 public:
  Node() {
    _value = 0;
    _id = ++__NODE_ID_IDENTITY;
  }
  void SetValue(float value) {
    if (value <= 1 && value >= 0)
      _value = value;
    else
      throw runtime_error("try to set a incorrect value in a node");
  }
  float GetValue() { return _value; }
  void printNode() { printf("[NODE:%d]:%.5f\n", _id, _value); }
};

class NNode : public Node {
 private:
  vector<float> _neuralWeights;
  int _neuralInputNumber;
  void ModifyWeight(float intensity, int index);

 public:
  NNode(int neuralInputs);
  void CalculateNode(vector<NNode> prevNodes);
  vector<float>* TrainNode(vector<NNode>& prevNodes, float desideredValues,
                           bool verbose);
  void printNode();
};

#endif