import { TRAINING_DATA, TESTING_DATA } from './load.js'
import NeuralNetwork from '../JS/VNN.js'
import path from 'path';
import { fileURLToPath } from 'url';

const nn = new NeuralNetwork(
    [TRAINING_DATA.inputs[0].length, 6, 4, TRAINING_DATA.labels[0].length],
    0.03,
    10000);

// nn.trainSGD(TRAINING_DATA);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
await nn.loadWB(path.join(__dirname, 'model.json'));
// nn.dumpWB(path.join(__dirname, 'model.json'));

let correct = 0;
let totalTest = TESTING_DATA.inputs.length;

for (let i = 0; i < totalTest; i++) {
    const inputArr = TESTING_DATA.inputs[i];
    const labelArr = TESTING_DATA.labels[i];
    const prediction = nn.predict(inputArr);
    const error = prediction.map((p, i) => (Math.pow(p - labelArr[i], 2)).toFixed('3'));
    console.log(`| INPUT: ${inputArr} | OUTPUT: ${prediction.map(x => x.toFixed('3'))} | LABEL: ${labelArr} | ERROR: ${error} |`);
    if (prediction.indexOf(Math.max(...prediction)) === labelArr.indexOf(Math.max(...labelArr)))
        correct++;
}
console.log("------- Result -------");
console.log("Correct: ", correct);
console.log("Total: ", totalTest);
console.log("Accuracy: ", (correct / totalTest * 100).toFixed('2'), '%');

