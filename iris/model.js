import DATA from './load.js'
import NeuralNetwork from '../JS/VNN.js'

const splitIndex = parseInt(DATA.inputs.length * 0.75);
const [TRAINING_INPUTS, TESTING_INPUTS] = [DATA.inputs.slice(0, splitIndex), DATA.inputs.slice(splitIndex,)];
const [TRAINING_LABELS, TESTING_LABELS] = [DATA.labels.slice(0, splitIndex), DATA.labels.slice(splitIndex,)];

const TRAINING_DATA = { inputs: TRAINING_INPUTS, labels: TRAINING_LABELS };
const TESTING_DATA = { inputs: TESTING_INPUTS, labels: TESTING_LABELS };

const nn = new NeuralNetwork(TRAINING_DATA.inputs[0].length, 5, TRAINING_DATA.labels[0].length, 0.003, 10000);

nn.train(TRAINING_DATA, 1);
// nn.eval(TESTING_DATA);

// await nn.loadWB();
// nn.dumpWB();

let correct = 0;
// let totalTest = TESTING_DATA.inputs.length;
let totalTest = TESTING_DATA.inputs.length;

for (let i = 0; i < totalTest; i++) {
    const inputArr = TESTING_DATA.inputs[i];
    const labelArr = TESTING_DATA.labels[i];
    const prediction = nn.predict(inputArr);
    // console.log(`\nInput: ${TESTING_DATA.inputs[i]}`)
    // console.log(`Predicted Class: ${prediction.map(v => v.toFixed('2'))}`);
    // console.log(`Actual Class   : ${TESTING_DATA.labels[0]}`);

    // console.log(`Predicted Class: ${prediction.indexOf(Math.max(...prediction))}`);
    // console.log(`Actual Class   : ${TESTING_DATA.labels[0].indexOf(Math.max(...TESTING_DATA.labels[0]))}`);

    if (prediction.indexOf(Math.max(...prediction)) === labelArr.indexOf(Math.max(...labelArr)))
        correct++;
}
console.log("------- Result -------");
console.log("Correct: ", correct);
console.log("Total: ", totalTest);
console.log("Accuracy: ", (correct / totalTest * 100).toFixed('2'), '%');

