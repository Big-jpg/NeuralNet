const NeuralNetwork = require('./VNN.js');

const data = {
    inputs: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ],
    labels: [
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ]
}

const nn = new NeuralNetwork(2, 2, 2, 0.3, 10000);
const confusionMatrix = [[0, 0], [0, 0]];
// row0: 
nn.train(data, 4);
console.log("\n|--------|---------|");
console.log("| INPUTS | OUTPUTS |");
for (let i = 0; i < data.inputs.length; i++) {
    let prediction = nn.predict(data.inputs[i])
    // console.log(`Input: ${data.inputs[i][0]} ,  ${data.inputs[i][1]} \t=> Output: ${prediction.map((x, i) => `${i}: ${x.toFixed('2')}`)}`);
    console.log(`|   ${data.inputs[i]}  |   ${prediction.map((x) => Math.round(x))}   |`);
    // console.log(`|   ${data.inputs[i]}  |    ${prediction.indexOf(Math.max(...prediction))}    |`);
    if (data.labels[i].indexOf(Math.max(...data.labels[i])) === prediction.indexOf(Math.max(...prediction))) {
        // console.log("TRUE")
        if (data.labels[i].indexOf(Math.max(...data.labels[i])) === 0) {
            confusionMatrix[0][0] += 1;
        } else {
            confusionMatrix[1][1] += 1;
        }
    } else {
        if (data.labels[i].indexOf(Math.max(...data.labels[i])) === 0) {
            confusionMatrix[0][1] += 1;
        } else {
            confusionMatrix[1][0] += 1;
        }
    }
}
console.log(`|ConfusionMatrix:`);
console.log(`| True Positive   | ${confusionMatrix[0][0]}`);
console.log(`| False Negative  | ${confusionMatrix[0][1]}`);
console.log(`| False Positive  | ${confusionMatrix[1][0]}`);
console.log(`| True Negative   | ${confusionMatrix[1][1]}`);
const prec = confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0])
const recl = confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[0][1])
console.log(`\n| Precision: ${prec}`);
console.log(`| Recall:    ${recl}`);
console.log(`| F1 score:  ${2 * prec * recl / (prec + recl)} `);
// console.log(`${2 * }`);

// nn.dumpWB();