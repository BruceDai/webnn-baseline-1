'use strict';

import {sizeOfShape, Tensor} from './tensor.js';

/**
 * Check the tensor whether it is a 1-D tensor and its length is equal to `expectedSize`.
 * @param {Tensor} a
 * @param {Number} expectedSize
 * @param {String} name
 */
function check1DTensorWithSize(a, expectedSize, name) {
  if (a) {
    if (a.rank !== 1) {
      throw new Error(`The parameter ${name} is not a 1-D tensor.`);
    } else {
      if (a.shape[0] !== expectedSize) {
        throw new Error(`The length ${a.shape[0]} of the ${name} values is not equal to the ` +
          `size ${expectedSize} of the input dimension denoted by options.axis.`);
      }
    }
  }
}

export function validateInput(op, args) {
  switch(op) {

  case "batchNormalization": {
    const [input, mean, variance, {axis=1, scale, bias, epsilon=1e-5, activation} = {}] = [...args];
    if (!Number.isInteger(axis)) {
      throw new Error(`Invalid axis ${axis}, axis should be an integer.`);
    }
    const dim = input.shape[axis];
    check1DTensorWithSize(mean, dim, 'mean');
    check1DTensorWithSize(variance, dim, 'variance');
    check1DTensorWithSize(scale, dim, 'scale');
    check1DTensorWithSize(bias, dim, 'bias');
  }
    break;

  case "concat": {
    const [inputs, axis] = [...args];
    const rank = inputs[0].rank;
    if (!Number.isInteger(axis)) {
      throw new Error(`Invalid axis ${axis}, axis should be an integer.`);
    } else {
      if (axis < 0 || axis >= rank) {
        throw new Error(`Invalid axis ${axis}, axis should be in the interval [0, ${rank}).`);
      }
    }
    const inputShape = inputs[0].shape;
    for (let i = 1; i < inputs.length; ++i) {
      if (inputs[i].rank !== rank) {
        throw new Error('All input tensors should have the same rank.');
      } else {
        const shape = inputs[i].shape;
        for (let j = 0; j < inputShape.length; ++j) {
          if (j !== axis) {
            if (inputShape[j] !== shape[j]) {
            throw new Error('All input tensors should have the same shape, ' +
              'except for the size of the dimension to concatenate on.');
            }
          }
        }
      }
    }
  }
    break;

  case "conv2": {
    const [input, filter, {bias, groups}] = [...args];
    const inputChannels = input.shape[1];
    const outputChannels = filter.shape[0];
    const filterInputChannels = filter.shape[1];
    if (input.rank !== 4) {
      throw new Error('The input should be a 4-D tensor.');
    }
    if (filter.rank !== 4) {
      throw new Error('The filter should be a 4-D tensor.');
    }
    if (inputChannels !== filterInputChannels * groups) {
      throw new Error('The input channels of filter is invalid.');
    }
    if (bias && (bias.rank !== 1 || bias.shape[0] != outputChannels)) {
      throw new Error('the bias should be a 1-D tensor with the shape of [output_channels].');
    }
  }
    break;

  case "gemm": {
    const [a, b] = [...args];
    if (a.rank !== 2) {
      throw new Error('The input a is not a 2-D tensor.');
    }
    if (b.rank !== 2) {
      throw new Error('The input b is not a 2-D tensor.');
    }
  }
    break;

  case "gruCell": {
    const [input, weight, recurrentWeight, hiddenState, hiddenSize,
           {bias, recurrentBias, resetAfter, layout = 'zrn', activations} = {}] = [...args];
    if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
      throw new Error(`The hiddenSize ${hiddenSize} is invalid.`);
    }
    if (input.rank !== 2) {
      throw new Error(`The input (rank ${input.rank}) is not a 2-D tensor.`);
    }
    const batchSize = input.shape[0];
    const inputSize = input.shape[1];
    if (weight.rank !== 2) {
      throw new Error(`The weight (rank ${weight.rank}) is not a 2-D tensor.`);
    }
    if (weight.shape[0] !== 3 * hiddenSize || weight.shape[1] !== inputSize) {
      throw new Error(`The shape of weight [${weight.shape[0]}, ${weight.shape[1]}] is invalid.`);
  }
    if (recurrentWeight.rank !== 2) {
      throw new Error(`The recurrentWeight (rank ${recurrentWeight.rank}) is not a 2-D tensor.`);
    }
    if (recurrentWeight.shape[0] !== 3 * hiddenSize || recurrentWeight.shape[1] !== hiddenSize) {
      throw new Error(`The shape of recurrentWeight ` +
      `[${recurrentWeight.shape[0]}, ${recurrentWeight.shape[1]}] is invalid.`);
    }
    if (hiddenState.rank !== 2) {
      throw new Error(`The hiddenState (rank ${hiddenState.rank}) is not a 2-D tensor.`);
    }
    if (hiddenState.shape[0] !== batchSize || hiddenState.shape[1] !== hiddenSize) {
      throw new Error(`The shape of hiddenState
      [${hiddenState.shape[0]}, ${hiddenState.shape[1]}] is invalid.`);
    }
    if (bias) {
      if (bias.rank !== 1) {
        throw new Error(`The bias (rank ${bias.rank}) is not a 1-D tensor.`);
      }
      if (bias.shape[0] !== 3 * hiddenSize) {
        throw new Error(`The shape of bias [${bias.shape[0]}] is invalid.`);
      }
    }
    if (recurrentBias) {
      if (recurrentBias.rank !== 1) {
        throw new Error(`The recurrentBias (rank ${bias.rank}) is not a 1-D tensor.`);
      }
      if (recurrentBias.shape[0] !== 3 * hiddenSize) {
        throw new Error(`The shape of recurrentBias [${recurrentBias.shape[0]}] is invalid.`);
      }
    }
    if (layout !== 'zrn' && layout !== 'rzn') {
      throw new Error(`The layout ${layout} is invalid.`);
    }
  }
    break;

  case "gru": {
    const [input, weight, recurrentWeight, steps, hiddenSize,
                    {bias, recurrentBias, initialHiddenState, resetAfter,
                     returnSequence, direction,
                     layout, activations}] = [...args];
    if (!Number.isInteger(steps) || steps <= 0) {
      throw new Error(`The steps ${steps} is invalid.`);
    }
    if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
      throw new Error(`The hiddenSize ${hiddenSize} is invalid.`);
    }
    if (input.rank !== 3) {
      throw new Error(`The input (rank ${input.rank}) is not a 3-D tensor.`);
    }
    if (input.shape[0] !== steps) {
      throw new Error(`The input.shape[0] ${input.shape[0]} is not equal to steps ${steps}.`);
    }
    const batchSize = input.shape[1];
    const inputSize = input.shape[2];
    if (direction !== 'forward' && direction !== 'backward' && direction !== 'both') {
      throw new Error(`The direction ${direction} is invalid.`);
    }
    const numDirections = (direction === 'both' ? 2 : 1);
    if (weight.rank !== 3) {
      throw new Error(`The weight (rank ${weight.rank}) is not a 3-D tensor.`);
    }
    if (weight.shape[0] !== numDirections || weight.shape[1] !== 3 * hiddenSize ||
        weight.shape[2] !== inputSize) {
      throw new Error(`The shape of weight [${weight.shape[0]}, ${weight.shape[1]},
        ${weight.shape[2]}] is invalid.`);
    }
    if (recurrentWeight.rank !== 3) {
      throw new Error(`The recurrentWeight (rank ${recurrentWeight.rank}) is not a 3-D tensor.`);
    }
    if (recurrentWeight.shape[0] !== numDirections || recurrentWeight.shape[1] !== 3 * hiddenSize ||
        recurrentWeight.shape[2] !== hiddenSize) {
      throw new Error(`The shape of recurrentWeight ` +
                      `[${recurrentWeight.shape[0]}, ${recurrentWeight.shape[1]}, ` +
                      `${recurrentWeight.shape[2]}] is invalid.`);
    }
    if (bias) {
      if (bias.rank !== 2) {
        throw new Error(`The bias (rank ${bias.rank}) is not a 2-D tensor.`);
      }
      if (bias.shape[0] !== numDirections || bias.shape[1] !== 3 * hiddenSize) {
        throw new Error(`The shape of bias [${bias.shape[0]}, ${bias.shape[1]}] is invalid.`);
      }
    }
    if (recurrentBias) {
      if (recurrentBias.rank !== 2) {
        throw new Error(`The recurrentBias (rank ${recurrentBias.rank}) is not a 2-D tensor.`);
      }
      if (recurrentBias.shape[0] !== numDirections || recurrentBias.shape[1] !== 3 * hiddenSize) {
        throw new Error(`The shape of recurrentBias [${recurrentBias.shape[0]},
          ${recurrentBias.shape[1]}] is invalid.`);
      }
    }
    let hiddenState;
    if (initialHiddenState) {
      if (initialHiddenState.rank !== 3) {
        throw new Error(
          `The initialHiddenState (rank ${initialHiddenState.rank}) is not a 3-D tensor.`);
      }
      if (initialHiddenState.shape[0] !== numDirections ||
          initialHiddenState.shape[1] !== batchSize ||
          initialHiddenState.shape[2] !== hiddenSize) {
        throw new Error(`The shape of initialHiddenState [${initialHiddenState.shape[0]},
          ${initialHiddenState.shape[1]}, ${initialHiddenState.shape[2]}] is invalid.`);
      }
      hiddenState = initialHiddenState;
    } else {
      const initialHiddenStateShape = [numDirections, batchSize, hiddenSize];
      hiddenState = new Tensor(
        initialHiddenStateShape, new Array(sizeOfShape(initialHiddenStateShape)).fill(0));
    }
    if (layout !== 'zrn' && layout !== 'rzn') {
      throw new Error(`The layout ${layout} is invalid.`);
    }
  }
    break;

  case "matmul": {
    const [a, b] = [...args];
    const aCols = a.shape[a.rank - 1];
    const bRows = b.shape[b.rank - 2];
    if (aCols !== bRows) {
      throw new Error(
        `The columns (${aCols}) of input a is not equal to rows (${bRows}) of input b.`);
    }
  }
    break;

  case "pool2d": {
    const [input, _, {roundingType = 'floor'}] = [...args];
    if (input.rank !== 4) {
      throw new Error('The input should be a 4-D tensor.');
    }
    if (roundingType !== 'floor' && roundingType !== 'ceil') {
      throw new Error('The rounding type is invalid.');
    }
  }
    break;

  case "reduce": {
    const [input, reduceFunc, {keepDimensions, axes}] = [...args];
    if (axes.length > input.rank) {
      throw new Error(`The length ${axes.length} of axes is bigger than input rank ${input.rank}.`);
    }
    for (let i = 0; i < axes.length; ++i) {
      if (axes[i] < 0 || axes[i] >= input.rank) {
        throw new Error(`The value ${axes[i]} at axis ${i} of axes is invalid.`);
      }
    }
  }
    break;

  case "slice": {
    const [input, starts, sizes, {axes} = {}] = [...args];
    let inpAxes = axes;
    const rank = input.rank;
    const startsForAllAxes = new Array(rank).fill(0);
    if (axes) {
      if (axes.length > rank) {
        throw new Error(`The length of axes ${axes.length} is greater than rank ${rank}.`);
      } else {
        for (const axis of axes) {
          if (!Number.isInteger(axis)) {
            throw new Error(`Invalid axes value ${axis}, it should be an integer.`);
          } else {
            if (axis >= rank || axis < -rank) {
              throw new Error(`Invalid axes value ${axis}, it should be in the interval ` +
                              `[${-rank}, ${rank}).`);
            }
          }
        }
      }
    } else {
      inpAxes = [...Array(rank).keys()];
    }
    const axesLen = inpAxes.length;
    if (starts.length !== axesLen) {
      throw new Error(`The length ${starts.length} of starts is not equal to the length ` +
                      `${axesLen} of axes.`);
    }
    if (sizes.length !== axesLen) {
      throw new Error(`The length ${sizes.length} of sizes is not equal to the length ${axesLen} ` +
                      'of axes.');
    }
    for (let i = 0; i < axesLen; ++i) {
      const axis = inpAxes[i] >= 0 ? inpAxes[i] : inpAxes[i] + rank;
      const size = input.shape[axis];
      const start = starts[i];
      if (!Number.isInteger(start)) {
        throw new Error(`Invalid starts value ${start}, it should be an integer.`);
    }
      startsForAllAxes[axis] = start >= 0 ? start : start + size;
      if (start >= size || start < -size) {
        throw new Error(`Invalid starts value ${start}, it shoule be in the interval ` +
                        `[${-size}, ${size}).`);
      } else {
        const sliceSize = sizes[i];
        if (!Number.isInteger(sliceSize)) {
          throw new Error(`Invalid sizes value ${sliceSize}, it should be an integer.`);
        }
        if (sliceSize >= 0) {
          if (start >= 0) {
            if (start + sliceSize > size) {
              throw new Error(`Invalid sizes value ${sliceSize}, the sum of the start ${start} ` +
                              `plus the size ${sliceSize} is greater than the dimensional size ${size}`);
            }
          } else {
            if (start + sliceSize > 0) {
              throw new Error(`Invalid sizes value ${sliceSize}, the sum of the start ${start} ` +
                              `plus the size ${sliceSize} is greater than the dimensional size ${size}`);
            }
          }
        } else {
          if (sliceSize !== -1) {
            throw new Error(`The value ${sliceSize} of sizes is invalid, it is required to be -1 ` +
                            'when it is negative.');
          }
        }
      }
    }
  }
    break;

  case "softmax": {
    const [x] = [...args];
    if (x.rank !== 2) {
      throw new Error('The input is not a 2-D tensor.');
    }
  }
    break;

  case "split": {
    const [input, splits, {axis = 0} = {}] = [...args];
    let inpAxis;
    if (axis !== undefined) {
      const rank = input.rank;
      if (!Number.isInteger(axis)) {
        throw new Error(`The axis ${axis} should be an integer.`);
      }
      if (axis >= rank || axis < -rank) {
        throw new Error(`The axis ${axis} should be in the interval [${-rank}, ${rank}).`);
      }
      inpAxis = axis >= 0 ? axis : rank + axis;
    }
    if (typeof splits === 'number') {
      if (!Number.isInteger(splits) || splits <= 0) {
        throw new Error(`Invalid splits ${splits}, it should be a positive integer.`);
      }
      if (input.shape[inpAxis] % splits !== 0) {
        throw new Error(`The splits ${splits} must evenly divide the dimension size ` +
                        `${input.shape[inpAxis]} of input along options.axis ${inpAxis}.`);
      }
    } else if (splits instanceof Array) {
      if (!splits.every((v) => Number.isInteger(v) && v > 0)) {
        throw new Error(`Invalid splits ${splits}, it should be an Array of positive integers.`);
      }
      const sum = splits.reduce((a, b) => a + b);
      if (sum !== input.shape[inpAxis]) {
        throw new Error(`Invalid [${splits}], the sum of sizes ${sum} must equal to the dimension ` +
                        `size ${input.shape[inpAxis]} of input along options.axis ${inpAxis}`);
      }
    }

  }
    break;

  case "squeeze": {
    const [input, {axes} = {}] = [...args];
    if (axes) {
      if (axes.length > input.rank) {
        throw new Error(`The length of axes ${axes.length} is bigger than input rank ${input.rank}.`);
      }

      for (const axis of axes) {
        if (axis < 0 || axis >= input.rank) {
          throw new Error(`The value of axes ${axis} is invalid.`);
        }
        if (axes && input.shape[axis] !== 1) {
          throw new Error(`The value ${input.shape[axis]} at axis ${axis} of input shape is not 1.`);
        }
      }
    }
  }
    break;

  case "tanh": {
    const [input, {permutation}] = [...args];
    if (permutation.length !== input.rank) {
      throw new Error(
        `The permutation length ${permutation.length} is not equal to rank ${input.rank}.`);
    }
  }
    break;
  }

}
