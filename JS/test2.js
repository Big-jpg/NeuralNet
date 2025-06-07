const NeuralNetwork = require('./VNN.js');

// const data = {
//     inputs: [
//         [0, 0, 0],
//         [0, 0, 1],
//         [0, 1, 0],
//         [0, 1, 1],
//         [1, 0, 0],
//         [1, 0, 1],
//         [1, 1, 0],
//         [1, 1, 1],
//     ],
//     labels: [
//         [0],
//         [1],
//         [1],
//         [0],
//         [1],
//         [0],
//         [0],
//         [1]
//     ]
// }
const data = {
    inputs: [
        [0, 0, 0],  // Sample 1
        [0, 0, 1],  // Sample 2
        [0, 1, 0],  // Sample 3
        [0, 1, 1],  // Sample 4
        [1, 0, 0],  // Sample 5
        [1, 0, 1],  // Sample 6
        [1, 1, 0],  // Sample 7
        [1, 1, 1],  // Sample 8
    ],
    labels: [
        [1, 0, 0],  // Class 0
        [0, 1, 0],  // Class 1
        [0, 1, 0],  // Class 1
        [1, 0, 0],  // Class 0
        [0, 0, 1],  // Class 2
        [1, 0, 0],  // Class 0
        [0, 0, 1],  // Class 2
        [0, 1, 0],  // Class 1
    ]
};

const nn = new NeuralNetwork(3, 5, 3, 0.2, 10000);
nn.train(data, 2);
for (let i = 0; i < data.inputs.length; i++) {
    let predictedOutput = nn.predict(data.inputs[i])
    console.log(`Input: ${data.inputs[i]} => Output: ${predictedOutput.map(x => x.toFixed('2'))}`);
}
// nn.dumpWB();