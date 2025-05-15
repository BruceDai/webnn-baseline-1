'use strict';

/* eslint guard-for-in: 0 */
import fs from 'fs';
import path from 'path';

import {argMax, argMin} from '../../src/arg_max_min.js';
import {batchNormalization} from '../../src/batch_normalization.js';
import {add, sub, mul, div, max, min, pow} from '../../src/binary.js';
import {cast} from '../../src/cast.js';
import {clamp} from '../../src/clamp.js';
import {concat} from '../../src/concat.js';
import {convTranspose2d} from '../../src/conv_transpose2d.js';
import {conv2d} from '../../src/conv2d.js';
import {cumulativeSum} from '../../src/cumulativeSum.js';
import {dequantizeLinear} from '../../src/dequantize_linear.js';
import {elu} from '../../src/elu.js';
import {expand} from '../../src/expand.js';
import {gatherElements} from '../../src/gather_elements.js';
import {gatherND} from '../../src/gather_nd.js';
import {gather} from '../../src/gather.js';
import {gelu} from '../../src/gelu.js';
import {gemm} from '../../src/gemm.js';
import {gruCell} from '../../src/gru_cell.js';
import {gru} from '../../src/gru.js';
import {hardSigmoid} from '../../src/hard_sigmoid.js';
import {hardSwish} from '../../src/hard_swish.js';
import {instanceNormalization} from '../../src/instance_normalization.js';
import {layerNormalization} from '../../src/layer_normalization.js';
import {leakyRelu} from '../../src/leaky_relu.js';
import {linear} from '../../src/linear.js';
import {
  equal,
  greater,
  greaterOrEqual,
  lesser,
  lesserOrEqual,
  logicalAnd,
  logicalNot,
  logicalOr,
  logicalXor,
  notEqual,
} from '../../src/logical.js';
import {lstmCell} from '../../src/lstm_cell.js';
import {lstm} from '../../src/lstm.js';
import {matmul} from '../../src/matmul.js';
import {pad} from '../../src/pad.js';
import {averagePool2d, l2Pool2d, maxPool2d} from '../../src/pool2d.js';
import {prelu} from '../../src/prelu.js';
import {quantizeLinear} from '../../src/quantize_linear.js';
import {
  reduceL1,
  reduceL2,
  reduceLogSum,
  reduceLogSumExp,
  reduceMax,
  reduceMean,
  reduceMin,
  reduceProduct,
  reduceSum,
  reduceSumSquare,
} from '../../src/reduce.js';
import {relu} from '../../src/relu.js';
import {resample2d} from '../../src/resample2d.js';
import {reshape} from '../../src/reshape.js';
import {reverse} from '../../src/reverse.js';
import {scatterElements} from '../../src/scatter_elements.js';
import {scatterND} from '../../src/scatter_nd.js';
import {sigmoid} from '../../src/sigmoid.js';
import {slice} from '../../src/slice.js';
import {softmax} from '../../src/softmax.js';
import {softplus} from '../../src/softplus.js';
import {softsign} from '../../src/softsign.js';
import {split} from '../../src/split.js';
import {tanh} from '../../src/tanh.js';
import {tile} from '../../src/tile.js';
import {transpose} from '../../src/transpose.js';
import {triangular} from '../../src/triangular.js';
import {
  abs,
  ceil,
  cos,
  exp,
  floor,
  log,
  neg,
  sin,
  tan,
  identity,
  reciprocal,
  sqrt,
  erf,
  sign,
} from '../../src/unary.js';
import {where} from '../../src/where.js';
import {Tensor, sizeOfShape} from '../../src/lib/tensor.js';

import {utils} from './utils.js';

/**
 * Container object for storing generated test input and output data.
 * Inputs are keyed by input data name; outputs are keyed by `${testName}${outputName}`.
 * @type {{inputs: Object<string, TypedArray>, outputs: Object<string, TypedArray>}}
 */
const testData = {
  inputs: {},
  outputs: {},
};
const testDataFolder = path.join(utils.currentFolder, 'test-data');

const operatorDictionary = {
  argMax,
  argMin,
  batchNormalization,

  // Element-wise binary operations
  add,
  sub,
  mul,
  div,
  max,
  min,
  pow,

  cast,
  clamp,
  concat,
  convTranspose2d,
  conv2d,
  cumulativeSum,
  dequantizeLinear,
  elu,
  expand,
  gatherElements,
  gatherND,
  gather,
  gelu,
  gemm,
  gruCell,
  gru,
  hardSigmoid,
  hardSwish,
  instanceNormalization,
  layerNormalization,
  leakyRelu,
  linear,

  // Element-wise logical operations
  equal,
  greater,
  greaterOrEqual,
  lesser,
  lesserOrEqual,
  logicalAnd,
  logicalNot,
  logicalOr,
  logicalXor,
  notEqual,

  lstmCell,
  lstm,
  matmul,
  pad,

  // Pooling operations
  averagePool2d,
  l2Pool2d,
  maxPool2d,

  prelu,
  quantizeLinear,

  // Reduction operations
  reduceL1,
  reduceL2,
  reduceLogSum,
  reduceLogSumExp,
  reduceMax,
  reduceMean,
  reduceMin,
  reduceProduct,
  reduceSum,
  reduceSumSquare,

  relu,
  resample2d,
  reshape,
  reverse,
  scatterElements,
  scatterND,
  sigmoid,
  slice,
  softmax,
  softplus,
  softsign,
  split,
  tanh,
  tile,
  transpose,
  triangular,

  // Element-wise unary operations
  abs,
  ceil,
  cos,
  exp,
  floor,
  log,
  neg,
  sin,
  tan,
  identity,
  reciprocal,
  sqrt,
  erf,
  sign,

  where,
};

/**
 * Generates input data for a graph's input resources using the test definitions.
 * Avoids regenerating data if it already exists in `testData.inputs`.
 *
 * @param {Object} resources - Input resource definitions from the test graph.
 */
function generateInputsData(resources) {
  for (const inputName in resources) {
    let inputDataName;
    if (typeof resources[inputName].data === 'string') {
      inputDataName = resources[inputName].data;
    } else if (
      typeof resources[inputName].data === 'object' &&
      resources[inputName].data.name
    ) {
      inputDataName = resources[inputName].data.name;
    }
    if (inputDataName) {
      if (!Object.getOwnPropertyDescriptor(testData.inputs, inputDataName)) {
        utils.generateData(resources[inputName], testData.inputs);
      } else {
        console.log(
            `No need generate data for ${inputDataName} since data already exsited`,
        );
      }
    } else {
      console.log(`No need generate data for ${inputName}`);
    }
  }
}

/**
 * Creates a Tensor instance from a resource descriptor and input data.
 *
 * @param {Object} resource - Resource object that describes tensor shape and source data.
 * @return {Tensor} - A Tensor instance populated with input data.
 */
function createInputTensor(resource) {
  const shape = resource.descriptor.shape;
  let tensorData = resource.data;
  if (typeof resource.data === 'string') {
    tensorData = testData.inputs[resource.data];
  } else if (typeof resource.data === 'object' && resource.data.name) {
    tensorData = testData.inputs[resource.data.name];
  }
  return new Tensor(
      shape,
      utils.getTypedArrayData(
          resource.descriptor.dataType,
          sizeOfShape(shape),
          tensorData,
      ),
  );
}

/**
 * Executes an operator graph for a test case to produce outputs.
 * Resolves all operator inputs (from graph inputs or previous results),
 * runs the operators, and stores output tensors in `testData.outputs`.
 *
 * @param {Object} resources - The full test case object including graph resources.
 */
function generateTestData(resources) {
  const testName = resources.name;
  console.log(`Start to generate test data for the test: ${testName}`);
  const graphResources = resources.graph;
  const graphInputs = graphResources.inputs;
  generateInputsData(graphInputs);

  const graphOperators = graphResources.operators;
  const intermediateOperands = {};
  for (const operator of graphOperators) {
    const argumentArray = [];
    for (const argument of operator.arguments) {
      for (const argumentName in argument) {
        if (argumentName !== 'options') {
          if (Object.getOwnPropertyDescriptor(graphInputs, argument[argumentName])) {
            const operandName = argument[argumentName];
            const operand = createInputTensor(graphInputs[operandName]);
            argumentArray.push(operand);
          } else if (
            Object.getOwnPropertyDescriptor(intermediateOperands, argument[argumentName])
          ) {
            argumentArray.push(intermediateOperands[argument[argumentName]]);
          } else {
            argumentArray.push(argument[argumentName]);
          }
        } else {
          for (const [optionalArgumentName, value] of Object.entries(
              argument['options'],
          )) {
            if (
              typeof value === 'string' &&
              !!Object.getOwnPropertyDescriptor(graphInputs, value)
            ) {
              const operandName = value;
              const operand = createInputTensor(graphInputs[operandName]);
              argument['options'][optionalArgumentName] = operand;
            } else if (
              typeof value === 'string' &&
              !!Object.getOwnPropertyDescriptor(intermediateOperands, value)
            ) {
              argument['options'][optionalArgumentName] =
                intermediateOperands[value];
            }
          }
          argumentArray.push(argument['options']);
        }
      }
    }

    const currentOutput = operatorDictionary[operator.name](...argumentArray);
    if (Array.isArray(operator.outputs)) {
      operator.outputs.forEach((outputName, index) => {
        intermediateOperands[outputName] = currentOutput[index];
      });
    } else {
      intermediateOperands[operator.outputs] = currentOutput;
    }
  }

  const outputNames = Object.keys(graphResources.expectedOutputs);
  outputNames.forEach((outputName) => {
    if (Object.getOwnPropertyDescriptor(intermediateOperands, outputName)) {
      const outputDataName = `${testName}${graphResources.expectedOutputs[outputName].data}`;
      if (!Object.getOwnPropertyDescriptor(testData.outputs, outputDataName)) {
        const outputTensor = intermediateOperands[outputName];
        testData.outputs[outputDataName] = utils.getTypedArrayData(
            graphResources.expectedOutputs[outputName].descriptor.dataType,
            sizeOfShape(outputTensor.shape),
            outputTensor.data,
        );
      }
    }
  });
}


function reuseExistedInputTestData(operatorName) {
  const testDataFile = path.join(testDataFolder, `${operatorName}.json`);
  if (fs.existsSync(testDataFile)) {
    const existedTestData = utils.readJsonFile(testDataFile);
    testData.inputs = existedTestData.inputs;
  }
}

/**
 * Saves all generated input and output data to a JSON file under `test-data/`.
 * Useful for debugging or manual verification of test data.
 *
 * @param {string} operatorName - The operator name used to name the output file.
 */
function saveTestDataFile(operatorName) {
  // Save test data into local file for manually updating input data if needed
  utils.mkdirIfNotExists(testDataFolder);
  const fileName = path.join(testDataFolder, `${operatorName}.json`);
  utils.writeJsonFile(testData, fileName);
}

/**
 * Rewrites the test definitions to inline the generated input and output data.
 * Outputs the result to a WPT-compatible JSON file under `wpt-test/`.
 *
 * @param {Object} resources - The full test suite including all test cases.
 * @param {string} operatorName - The operator name used to name the output file.
 */
function saveWPTTestFile(resources, operatorName) {
  const wptTestFolder = path.join(utils.currentFolder, 'wpt-test');
  utils.mkdirIfNotExists(wptTestFolder);
  const fileName = path.join(wptTestFolder, `${operatorName}.json`);
  const updatedResources = {tests: []};

  resources.tests.forEach((test) => {
    const graphResources = test.graph;
    // update inputs data
    for (const inputName in graphResources.inputs) {
      // 1. the value of data is a string name like below
      // {
      //   "data": "float32InputData2D",
      //   "descriptor": {"shape": [3, 3], "dataType": "float32"}
      // },

      // 2. the value of data is a object like below
      // {
      //   "data": {
      //     "name": "float32InputData2DT",
      //     "sourceData": {
      //       "name": "float32InputData2D",
      //       "process": "transpose", // transpose, negative
      //       "permutation": [1, 0] // permutation option only for transpose
      //     },
      //     "specified": { // optional
      //       "dataRange": {
      //          "max": 10,
      //          "min": -10
      //       },
      //       "sign": 'mixed' // // mixed (default), positive, negative
      //     }
      //   },
      //   "descriptor": {"shape": [3, 3], "dataType": "float32"}
      // },
      const inputDescriptor = graphResources.inputs[inputName].descriptor;
      let inputData;
      if (typeof graphResources.inputs[inputName].data === 'string') {
        inputData =
          utils.getTypedArrayData(
              inputDescriptor.dataType,
              sizeOfShape(inputDescriptor.shape),
              testData.inputs[graphResources.inputs[inputName].data],
          );
        graphResources.inputs[inputName].data = inputData;
      } else if (typeof graphResources.inputs[inputName].data === 'object') {
        if (graphResources.inputs[inputName].data.name) {
          inputData = utils.getTypedArrayData(
              inputDescriptor.dataType,
              sizeOfShape(inputDescriptor.shape),
              testData.inputs[graphResources.inputs[inputName].data.name],
          );
          graphResources.inputs[inputName].data = inputData;
        }
      }
    }
    // update expected data
    for (const outputName in graphResources.expectedOutputs) {
      graphResources.expectedOutputs[outputName].data =
        testData.outputs[
            test.name + graphResources.expectedOutputs[outputName].data
        ];
    }
    updatedResources.tests.push(test);
  });

  utils.writeJsonFile(updatedResources, fileName);
}

/**
 * Reads test definitions, generates data, executes the graph,
 * and saves both test data and updated WPT tests to disk.
 */
function main() {
  const rawTestFile = process.argv[2];
  const operatorName = path.basename(rawTestFile, '.json');
  const testResources = utils.readJsonFile(rawTestFile);

  reuseExistedInputTestData(operatorName);

  testResources.tests.forEach((test) => {
    generateTestData(test);
  });

  saveTestDataFile(operatorName);
  saveWPTTestFile(testResources, operatorName);
}

main();
