const fs = require('fs');
const path = require('path');

const filePath = path.resolve('./iris.data');
// console.log(filePath)
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

// console.log(DATA.inputs[149]); // 0 to 149

for (let i = 0; i < DATA.inputs.length; i++) {
    const rand = parseInt(Math.random() * DATA.inputs.length) % DATA.inputs.length;
    swap(DATA.inputs, DATA.labels, i, rand);
}

// for (let i = 0; i < 15; i++) {
//     console.log(DATA.inputs[i], DATA.labels[i]);
// }

module.exports = DATA;