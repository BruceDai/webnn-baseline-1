'use strict';

/* eslint guard-for-in: 0 */

import path from 'path';
import {convTranspose2d} from '../../src/conv_transpose2d.js';
import {clamp} from '../../src/clamp.js';
import {leakyRelu} from '../../src/leaky_relu.js';
import {relu} from '../../src/relu.js';
import {sigmoid} from '../../src/sigmoid.js';
import {Tensor} from '../../src/lib/tensor.js';
import {utils} from './utils.js';

(() => {
  function convTranspose2dCompute(
      input, filter, options = {}, activationOptions = {}) {
    const inputTensor = new Tensor(input.shape, input.data);
    const filterTensor = new Tensor(filter.shape, filter.data);
    if (options && options.bias) {
      options.bias = new Tensor(options.bias.shape, options.bias.data);
    }
    if (options && options.activation === 'relu') {
      options.activation = relu;
    } else if (options && options.activation === 'relu6') {
      options.activation =
          utils.bindTrailingArgs(clamp, {minValue: 0, maxValue: 6});
    } else if (options && options.activation === 'sigmoid') {
      options.activation = sigmoid;
    } else if (options && options.activation === 'leakyRelu') {
      options.activation =
          utils.bindTrailingArgs(leakyRelu, activationOptions);
    }
    const outputTensor = convTranspose2d(inputTensor, filterTensor, options);
    return outputTensor.data;
  }

  const savedDataFile = path.join(
      path.dirname(process.argv[1]), 'test-data', 'conv-transpose2d-data.json');
  const jsonDict = utils.readJsonFile(process.argv[2]);
  const inputsDataInfo = jsonDict.inputsData;
  const inputsDataRange = jsonDict.inputsDataRange;
  const toSaveDataDict = utils.prepareInputsData(
      inputsDataInfo, savedDataFile, inputsDataRange.min, inputsDataRange.max);
  toSaveDataDict['expectedData'] = {};
  const tests = jsonDict.tests;
  const wptTests = JSON.parse(JSON.stringify(tests));
  for (const test of tests) {
    const precisionDataInput = utils.getPrecisionDataFromDataDict(
        toSaveDataDict['inputsData'], test.inputs.input.data,
        test.inputs.input.type);
    const input = {shape: test.inputs.input.shape, data: precisionDataInput};
    const precisionDataFilter = utils.getPrecisionDataFromDataDict(
        toSaveDataDict['inputsData'], test.inputs.filter.data,
        test.inputs.filter.type);
    const filter = {shape: test.inputs.filter.shape, data: precisionDataFilter};
    if (test.options && test.options.bias) {
      test.options.bias['data'] = utils.getPrecisionDataFromDataDict(
          toSaveDataDict['inputsData'], test.options.bias.data,
          test.options.bias.type);
    }
    const result = convTranspose2dCompute(input, filter, test.options);
    toSaveDataDict['expectedData'][test.expected.data] =
      utils.getPrecisionData(result, test.expected.type);
  }

  utils.writeJsonFile(toSaveDataDict, savedDataFile);
  console.log(`[ Done ] Saved test data into ${savedDataFile}.`);

  const wptConformanceTestsDict = {tests: []};
  for (const test of wptTests) {
    // update inputs data
    for (const inputName in test.inputs) {
      test.inputs[inputName].data =
          typeof test.inputs[inputName].data === 'number' ||
          (typeof test.inputs[inputName].data === 'object' &&
           typeof test.inputs[inputName].data[0] === 'number') ?
          test.inputs[inputName].data :
          toSaveDataDict['inputsData'][test.inputs[inputName].data];
    }
    // update weights (scale, bias, and etc.) data of options
    if (test.options) {
      for (const optionName in test.options) {
        if (test.options[optionName].data) {
          test.options[optionName].data =
              toSaveDataDict['inputsData'][test.options[optionName].data];
        }
      }
    }
    // update expected data
    test.expected.data = toSaveDataDict['expectedData'][test.expected.data];
    wptConformanceTestsDict.tests.push(test);
  }
  const savedWPTDataFile = path.join(
      path.dirname(process.argv[1]), 'test-data-wpt', 'conv_transpose2d.json');
  utils.writeJsonFile(wptConformanceTestsDict, savedWPTDataFile);

  console.log(`[ Done ] Generate test data file for WPT tests.`);
})();
