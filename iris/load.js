const fs = require('fs');
const path = require('path');

const filePath = path.resolve('./iris.data');
const fileData = fs.readFileSync(filePath, 'utf8');
const rows = fileData.split('\n');
const data = rows.map(r => {
    return r.split(',').map(item => item.trim());
});

const DATA = { inputs: [], labels: [] };

data.map((r, i) => {
    const inputs = r.slice(0, -1);
    const cls = r.slice(-1)[0];
    const label = new Array(3).fill(0);
    switch (cls) {
        case "Iris-setosa":
            label[0] = 1;
            break;
        case "Iris-versicolor":
            label[1] = 1;
            break;
        case "Iris-virginica":
            label[2] = 1;
            break;
        default:
            break;
    }
    DATA.inputs[i] = inputs;
    DATA.labels[i] = label;
})

function swap(arr1, arr2, i, rand) {
    [arr1[i], arr1[rand]] = [arr1[rand], arr1[i]];
    [arr2[i], arr2[rand]] = [arr2[rand], arr2[i]];
}

for (let i = 0; i < DATA.inputs.length; i++) {
    const rand = parseInt(Math.random() * DATA.inputs.length) % DATA.inputs.length;
    swap(DATA.inputs, DATA.labels, i, rand);
}

const splitIndex = parseInt(DATA.inputs.length * 0.75);
const [TRAINING_INPUTS, TESTING_INPUTS] = [DATA.inputs.slice(0, splitIndex), DATA.inputs.slice(splitIndex,)];
const [TRAINING_LABELS, TESTING_LABELS] = [DATA.labels.slice(0, splitIndex), DATA.labels.slice(splitIndex,)];

const TRAINING_DATA = { inputs: TRAINING_INPUTS, labels: TRAINING_LABELS };
const TESTING_DATA = { inputs: TESTING_INPUTS, labels: TESTING_LABELS };

module.exports = { TRAINING_DATA, TESTING_DATA };