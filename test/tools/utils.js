'use strict';

import fs from 'fs';
import path from 'path';
import {Float16Array} from '@petamoriken/float16';

import {Tensor, sizeOfShape} from '../../src/lib/tensor.js';
import {neg} from '../../src/unary.js';
import {transpose} from '../../src/transpose.js';

const currentFolder = path.dirname(process.argv[1]);

/**
 * Default valid data ranges for each supported data type.
 */
const defaultDataRange = {
  int8: {
    min: -128,
    max: 127,
  },
  uint8: {
    min: 0,
    max: 255,
  },
  int32: {
    min: -Math.pow(2, 31),
    max: Math.pow(2, 31) - 1,
  },
  uint32: {
    min: 0,
    max: Math.pow(2, 32) - 1,
  },
  int64: {
    // int64 range: [/* -(2**63) */ –9223372036854775808, /* 2**63 - 1 */ 92233720368547758087]
    // The maximum safe integer in JavaScript (2**53 - 1)
    // The minimum safe integer in JavaScript -(2**53 - 1)
    min: -(Math.pow(2, 53) - 1),
    max: Math.pow(2, 53) - 1,
  },
  uint64: {
    // uint64 range: [0, /* 2**64 - 1 */ 18446744073709551615]
    // The maximum safe integer in JavaScript (2**53 - 1)
    min: 0,
    max: Math.pow(2, 53) - 1,
  },
  float16: {
    // https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    min: -65504, // 1 11110 1111111111
    max: 65504, // 0 11110 1111111111
  },
  float32: {
    // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
    // largest normal number (0 11111110 11111111111111111111111): 2**127 * [2 − 2**(−23)]
    // The maximum safe integer in JavaScript (2**53 - 1)
    // The minimum safe integer in JavaScript -(2**53 - 1)
    min: -(Math.pow(2, 53) - 1),
    max: Math.pow(2, 53) - 1,
  },
};

/**
 * Map of supported data types to corresponding TypedArray constructors.
 */
const TypedArrayDict = {
  // https://www.w3.org/TR/webnn/#enumdef-mloperanddatatype
  float32: Float32Array,
  float16: Float16Array,
  int32: Int32Array,
  uint32: Uint32Array,

  // TODO: support int64 and uint64
  // current using Int32Array for int64 and Uint32Array for uint64
  // int64: BigInt64Array,
  // uint64: BigUint64Array,
  int64: Int32Array,
  uint64: Uint32Array,

  int8: Int8Array,
  uint8: Uint8Array,

  int4: Uint8Array, // Packed into Uint8Array
  uint4: Uint8Array, // Packed into Uint8Array
};

/**
 * Convert an input number or array to a typed array of the given data type.
 * @param {Array<Number>|Number} input - The number(s) to convert.
 * @param {String} dataType - The target data type (e.g., 'float32', 'int64').
 * @return {TypedArray|Number} - Converted data in specified precision.
 */
function getPrecisionData(input, dataType) {
  let data;
  const isNumber = typeof input === 'number';
  if (isNumber) {
    input = [input];
  }

  switch (dataType) {
    case 'float16':
      data = new Float16Array(input);
      break;
    case 'float32':
      data = new Float32Array(input);
      break;
    case 'int8':
      data = new Int8Array(input);
      break;
    case 'uint8':
      data = new Uint8Array(input);
      break;
    case 'int32':
      data = new Int32Array(input);
      break;
    case 'uint32':
      data = new Uint32Array(input);
      break;
    case 'int64':
      // TODO: support int64 and uint64
      // data = new BigInt64Array(input.map((x) => BigInt(x)));

      data = new Int32Array(input);
      break;
    case 'uint64':
      // TODO: support int64 and uint64
      // data = new BigUint64Array(input.map((x) => BigInt(x)));

      data = new Uint32Array(input);
      break;
    default:
      break;
  }

  if (isNumber) {
    data = data[0];
  }
  return data;
}

/**
 * Validate that a custom min/max range is within the default allowed limits.
 * @param {Number} min - Minimum value.
 * @param {Number} max - Maximum value.
 * @param {String} dataType - Data type to validate against.
 */
function validateMinMax(min, max, dataType) {
  if (min > max) {
    throw new Error(`The min should be lesser than max.`);
  }

  const defaultMin = defaultDataRange[dataType].min;
  const defaultMax = defaultDataRange[dataType].max;
  if (min < defaultMin || max > defaultMax) {
    throw new Error(
        `The range of ${dataType} type should be [${defaultMin}, ${defaultMax}].`,
    );
  }
}

/**
 * Get a random floating point number in [0, 1] inclusive.
 * @return {Number}
 */
function getFloatRandomInclusive() {
  // The Math.random() method returns a random floating point number in [0, 1),
  // below code would return a random floating point number in [0, 1].
  return Math.min(1, Math.random() + Number.EPSILON);
}

/**
 * Get a random integer between min and max (inclusive).
 * @param {Number} min
 * @param {Number} max
 * @return {Number}
 */
function getIntRandomInclusive(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Generate a random value based on dataType and optional constraints.
 * @param {string} dataType - The data type (e.g., 'float32', 'int32').
 * @param {Object} [resources={}] - Optional constraints for value generation.
 * @param {{min: number, max: number}} [resources.dataRange] - Min and max bounds.
 * @param {string} [resources.sign] - sign's value: 'mixed' (default), 'positive', or 'negative'.
 * @return {number|bigint} A randomly generated number or BigInt based on constraints.
 */
function getRandom(dataType, resources={}) {
  let min = defaultDataRange[dataType].min;
  let max = defaultDataRange[dataType].max;
  let data;

  const sign = resources?.sign || 'mixed';
  if (sign === 'positive') {
    min = 0;
  } else if (sign === 'negative') {
    max = 0;
  }

  if (resources?.dataRange) {
    min = resources.dataRange.min;
    max = resources.dataRange.max;
    validateMinMax(min, max, dataType);
  }

  if (dataType === 'float32' || dataType === 'float16') {
    const factor = getFloatRandomInclusive();
    data = factor * (max - min) + min;
  } else {
    // integer
    data = getIntRandomInclusive(min, max);

    // TODO: support int64 and uint64
    // if (['int64', 'uint64'].includes(dataType)) {
    //   data = BigInt(data);
    // }
  }

  return data;
}

/**
 * Convert raw data into a typed array of specified data type and size.
 * Special handling is included for int64/uint64 (BigInt) and int4/uint4 (bit-packed).
 * @param {String} dataType
 * @param {Number} size
 * @param {Array|Number} data
 * @return {TypedArray}
 */
function getTypedArrayData(dataType, size, data) {
  let outData;
  if (dataType === 'int64' || dataType === 'uint64') {
    if (typeof data === 'number' && size > 1) {
      return new TypedArrayDict[dataType](size).fill(BigInt(data));
    }
    outData = new TypedArrayDict[dataType](data.length);
    for (let i = 0; i < data.length; i++) {
      // TODO: support int64 and uint64
      // outData[i] = BigInt(data[i]);
      outData[i] = data[i];
    }
  } else if (dataType === 'uint4' || dataType === 'int4') {
    // The first nybble is stored in the first bits 0-3, and later bits 4-7
    // store the later nybble. The data is packed, without any padding between
    // dimensions. For example: an array of uint4:
    //   size = [2,5]
    //   values = [1,2,3,4,5,6,7,8,9,10]
    // Would yield 5 hex bytes:
    //   Uint8Array.of(0x21, 0x43, 0x65, 0x87, 0xA9);
    const array = new TypedArrayDict[dataType](Math.ceil(size / 2));
    let i = 0;
    while (i < size - 1) {
      const packedByte = ((data[i + 1] & 0xf) << 4) | (data[i] & 0xf);
      array[Math.floor(i / 2)] = packedByte;
      i = i + 2;
    }
    // Handle the odd size.
    if (i === size - 1) {
      const packedByte = data[i] & 0xf;
      array[Math.floor(i / 2)] = packedByte;
    }
    return array;
  } else {
    if (typeof data === 'number' && size > 1) {
      return new TypedArrayDict[dataType](size).fill(data);
    }
    outData = new TypedArrayDict[dataType](data);
  }
  return outData;
}

/**
 * Create a folder if it does not exist.
 * @param {String} folderName
 */
function mkdirIfNotExists(folderName) {
  if (!fs.existsSync(folderName)) {
    fs.mkdirSync(folderName);
  }
}

/**
 * Read a JSON file and return the parsed object.
 * Supports relative or absolute file paths. Strips comments from JSON.
 * @param {String} filePath
 * @return {Object}
 */
function readJsonFile(filePath) {
  let inputFile;
  if (path.isAbsolute(filePath)) {
    inputFile = filePath;
  } else {
    inputFile = path.join(currentFolder, filePath);
  }
  const content = fs.readFileSync(inputFile).toString();
  const jsonDict = JSON.parse(
      content.replace(
          /\\"|"(?:\\"|[^"])*"|(\/\/.*|\/\*[\s\S]*?\*\/)/g, // remove comments
          (m, g) => (g ? '' : m),
      ),
  );
  return jsonDict;
}

/**
 * Save a JavaScript object as a JSON file, converting TypedArrays into plain arrays.
 * @param {Object} jsonDict
 * @param {String} saveFile
 */
function writeJsonFile(jsonDict, saveFile) {
  const parentDirectory = path.dirname(saveFile);
  if (!fs.existsSync(parentDirectory)) {
    fs.mkdirSync(parentDirectory);
  }
  const jsonString = JSON.stringify(
      jsonDict,
      function(key, value) {
        // the replacer function is looking for some typed arrays.
        // If found, it replaces it by a trio
        if (
          value instanceof Int8Array ||
          value instanceof Uint8Array ||
          value instanceof Int32Array ||
          value instanceof Uint32Array ||
          value instanceof BigInt64Array ||
          value instanceof BigUint64Array ||
          value instanceof Float16Array ||
          value instanceof Float32Array
        ) {
          if (value.length === 1) {
            const result = [];
            result[0] = value[0];
            return result;
          } else {
            return Array.apply([], value);
          }
        }
        return value;
      },
      2,
  );
  fs.writeFileSync(saveFile, jsonString);
}

/**
 * Generate tensor data for testing.
 * Supports:
 * - Fresh random generation.
 * - Transformation (transpose/negate) of existing data.
 * - Constraints via data range and sign.
 * @param {Object} resources - Contains `descriptor` and `data` definition.
 * @param {Object} data - Shared data pool to populate.
 */
function generateData(resources, data) {
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
  const targetDataShape = resources.descriptor.shape;
  const targetDataType = resources.descriptor.dataType;
  let inputDataName;
  let targetData;

  if (typeof resources.data === 'string') {
    inputDataName = resources.data;
    if (!Object.getOwnPropertyDescriptor(data, inputDataName)) {
      const size = sizeOfShape(targetDataShape);
      targetData = new Array(size);
      for (let i = 0; i < size; i++) {
        targetData[i] = getRandom(targetDataType);
      }
      data[inputDataName] = getPrecisionData(targetData, targetDataType);
    }
  } else if (typeof resources.data === 'object' && resources.data.name) {
    // process source data for target data
    inputDataName = resources.data.name;
    if (!Object.getOwnPropertyDescriptor(data, inputDataName)) {
      if (resources.data.sourceData) {
        const sourceDataResources = resources.data.sourceData;
        const sourceData = data[sourceDataResources.name];
        let outTensor;
        if (sourceDataResources.process === 'transpose') {
          const permutation = sourceDataResources.permutation;
          const sourceDataShape = new Array(targetDataShape.length);
          for (let i = 0; i < targetDataShape.length; ++i) {
            sourceDataShape[permutation[i]] = targetDataShape[i];
          }
          const inputTensor = new Tensor(sourceDataShape, sourceData);
          outTensor = transpose(inputTensor, {permutation});
        } else if (sourceDataResources.process === 'negative') {
          const sourceDataShape = targetDataShape;
          const inputTensor = new Tensor(sourceDataShape, sourceData);
          outTensor = neg(inputTensor);
        }
        targetData = outTensor.data;
      } else {
        let specifiedResources = {};
        if (Object.getOwnPropertyDescriptor(resources.data, 'specified')) {
          specifiedResources = resources.data.specified;
        }
        const size = sizeOfShape(targetDataShape);
        targetData = new Array(size);
        for (let i = 0; i < size; i++) {
          targetData[i] = getRandom(targetDataType, specifiedResources);
        }
      }
      data[inputDataName] = getPrecisionData(targetData, targetDataType);
    }
  }
}

export const utils = {
  currentFolder, // Directory of the running script.
  generateData, // Main function to generate/transform tensor data.
  getPrecisionData, // Convert number(s) to specified typed array.
  getTypedArrayData, // Generate typed array from JS array.
  mkdirIfNotExists, // Create directory if missing.
  readJsonFile, // Load JSON (with comment stripping).
  writeJsonFile, // Save JSON (with TypedArray handling).
};
