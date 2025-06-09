import { train, test } from './load.js'
const DATA = { inputs: [...train.inputs, ...test.inputs], labels: [...train.labels, ...test.labels] }


function swap(arr1, arr2, i, rand) {
    [arr1[i], arr1[rand]] = [arr1[rand], arr1[i]];
    [arr2[i], arr2[rand]] = [arr2[rand], arr2[i]];
}

for (let i = 0; i < DATA.inputs.length; i++) {
    const rand = parseInt(Math.random() * DATA.inputs.length) % DATA.inputs.length;
    swap(DATA.inputs, DATA.labels, i, rand);
}

const train2 = {
    inputs: DATA.inputs.slice(0, 50000),
    labels: DATA.labels.slice(0, 50000)
}

const test2 = {
    inputs: DATA.inputs.slice(50000),
    labels: DATA.labels.slice(50000)
}

import NeuralNetwork from '../JS/VNN.js'
import path from 'path';
import { fileURLToPath } from 'url';

train2.inputs = train2.inputs.map(inp => {
    return inp.map(x => x / 255.0);
})

test2.inputs = test2.inputs.map(inp => {
    return inp.map(x => x / 255.0);
})

const nn = new NeuralNetwork(
    [train2.inputs[0].length, 16, 16, train2.labels[0].length],
    0.01,
    10);

nn.trainSGD(train2);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// await nn.loadWB(path.join(__dirname, 'model.json'));
nn.dumpWB(path.join(__dirname, 'model.json'));

let correct = 0;
let totalTest = Math.min(test2.labels.length, test2.inputs.length);

for (let i = 0; i < totalTest; i++) {
    const inputArr = test2.inputs[i];
    const labelArr = test2.labels[i];
    const prediction = nn.predict(test2.inputs[i]);
    const error = prediction.map((p, i) => (Math.pow(p - labelArr[i], 2)).toFixed('3'));

    // console.log(`| INPUT: ${inputArr} | OUTPUT: ${prediction.map(x => x.toFixed('3'))} | LABEL: ${labelArr} | ERROR: ${error} |`);
    // console.log(`| OUTPUT: ${prediction.map(x => x.toFixed('3'))} |\n| LABEL: ${labelArr} |\n| ERROR: ${error} |`);

    if (prediction.indexOf(Math.max(...prediction)) === labelArr.indexOf(Math.max(...labelArr)))
        correct++;
}
console.log("------- Result -------");
console.log("Correct: ", correct);
console.log("Total: ", totalTest);
console.log("Accuracy: ", (correct / totalTest * 100).toFixed('2'), '%');

