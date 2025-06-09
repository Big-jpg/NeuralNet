import { train, test } from './load.js'
const DATA = { inputs: [...train.inputs, ...test.inputs], labels: [...train.labels, ...test.labels] }
import NeuralNetwork from '../JS/VNN.js'
import path from 'path';
import { fileURLToPath } from 'url';


function swap(arr1, arr2, i, rand) {
    [arr1[i], arr1[rand]] = [arr1[rand], arr1[i]];
    [arr2[i], arr2[rand]] = [arr2[rand], arr2[i]];
}

for (let i = 0; i < DATA.inputs.length; i++) {
    const rand = parseInt(Math.random() * DATA.inputs.length) % DATA.inputs.length;
    swap(DATA.inputs, DATA.labels, i, rand);
}

const train2 = {
    inputs: DATA.inputs.slice(0, 60000),
    labels: DATA.labels.slice(0, 60000)
}

const test2 = {
    inputs: DATA.inputs.slice(60000),
    labels: DATA.labels.slice(60000)
}

train2.inputs = train2.inputs.map(inp => {
    return inp.map(x => x / 255.0);
})

test2.inputs = test2.inputs.map(inp => {
    return inp.map(x => x / 255.0);
})

const nn = new NeuralNetwork(
    [train2.inputs[0].length, 18, 16, train2.labels[0].length],
    0.01,
    20);

nn.trainSGD(train2);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// await nn.loadWB(path.join(__dirname, 'WebModel02.json'));
nn.dumpWB(path.join(__dirname, 'WebModel04.json'));

nn.evaluate(test2);
// const drawPred = nn.predict(draw);
// console.log("This is number:", drawPred.indexOf(Math.max(...drawPred)));