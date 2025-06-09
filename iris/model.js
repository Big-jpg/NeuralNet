import { TRAINING_DATA, TESTING_DATA } from './load.js'
import NeuralNetwork from '../JS/VNN.js'
import path from 'path';
import { fileURLToPath } from 'url';

const nn = new NeuralNetwork(
    [TRAINING_DATA.inputs[0].length, 3, 3, TRAINING_DATA.labels[0].length],
    0.01,
    10000);

nn.trainSGD(TRAINING_DATA);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// await nn.loadWB(path.join(__dirname, 'model.json'));
nn.dumpWB(path.join(__dirname, 'model.json'));

let correct = 0;
let totalTest = TESTING_DATA.inputs.length;

for (let i = 0; i < totalTest; i++) {
    const inputArr = TESTING_DATA.inputs[i];
    const labelArr = TESTING_DATA.labels[i];
    const prediction = nn.predict(inputArr);
    // console.log(`| INPUT: ${inputArr} | OUTPUT: ${prediction} | LABEL: ${labelArr} |`);
    if (prediction.indexOf(Math.max(...prediction)) === labelArr.indexOf(Math.max(...labelArr)))
        correct++;
}
console.log("------- Result -------");
console.log("Correct: ", correct);
console.log("Total: ", totalTest);
console.log("Accuracy: ", (correct / totalTest * 100).toFixed('2'), '%');

