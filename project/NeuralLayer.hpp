#ifndef LAYER
#define LAYER
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include "Configuration.hpp"
#include "NeuralNode.hpp"

using namespace std;

class NeuralLayer {
 protected:
  vector<NeuralNode> _nodes;
  int _nodeNum, _layerId;

 public:
  NeuralLayer(int nodeNumber, int prevLayerNode, int layerId) {
    _layerId = layerId;
    _nodeNum = nodeNumber;
    for (int ind = 0; ind < _nodeNum; ind++) {
      _nodes.push_back(NeuralNode(prevLayerNode));
    }
  }
  void PrintLayer() {
    printf("[[LAYER %d]]\n", _layerId);
    for (NeuralNode n : _nodes) {
      n.printNode();
    }
  }
  vector<NeuralNode>& GetNodes() { return _nodes; }
  int GetNodeNum() { return _nodeNum; }
  void CalculateLayer(vector<NeuralNode>& prevNodes) {
    for (NeuralNode& n : _nodes) {
      n.CalculateNode(prevNodes);
    }
  }
  vector<float> TrainLayer(vector<NeuralNode>& prevNodes,
                           vector<float>& desideredValues, bool verbose) {
    vector<float> prevLayerDesideredValue;
    vector<float>* changes;
    if (verbose) printf("[[TRAIN LAYER %d]]\n", _layerId);
    prevLayerDesideredValue.resize(prevNodes.size(), 0);
    for (int i = 0; i < _nodes.size(); i++) {
      changes = _nodes[i].TrainNode(prevNodes, desideredValues[i], verbose);
      for (int i = 0; i < prevNodes.size(); i++) {
        prevLayerDesideredValue[i] += (*changes)[i];
      }
    }
    delete changes;
    for (int i = 0; i < prevNodes.size(); i++) {
      prevLayerDesideredValue[i] =
          prevNodes[i].GetValue() +
          prevLayerDesideredValue[i] * PARAM_BACK_PROPAGATION;
      if (prevLayerDesideredValue[i] > 1) prevLayerDesideredValue[i] = 1;
      if (prevLayerDesideredValue[i] < 0) prevLayerDesideredValue[i] = 0;
    }
    if (verbose) printf("[[TRAIN LAYER END]]\n");
    return prevLayerDesideredValue;
  }
};

class InputNeuralLayer : public NeuralLayer {
 public:
  InputNeuralLayer(int nodeNumber, int layerId)
      : NeuralLayer(nodeNumber, 0, layerId) {}
  void SetValues(vector<float> inputValue) {
    if (inputValue.size() != _nodeNum)
      throw runtime_error("[InputNeuralLayer] error in insert input value");
    for (int ind = 0; ind < _nodeNum; ind++) {
      _nodes[ind].SetValue(inputValue[ind]);
    }
  }
  void PrintLayer() {
    printf("[[INITIAL_LAYER]]\n");
    for (Node& n : _nodes) {
      n.printNode();
    }
  }
};

#endif  // LAYER