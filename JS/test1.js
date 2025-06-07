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

const nn = new NeuralNetwork(2, 2, 2, 0.1, 10000);

nn.train(data, 1);
for (let i = 0; i < data.inputs.length; i++) {
    let Oo = nn.predict(data.inputs[i])
    console.log(`Input: ${data.inputs[i]} => Output: ${Oo.map((x, i) => `${i}: ${x.toFixed('2')}`)}`);
}
// nn.dumpWB();