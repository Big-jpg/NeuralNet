const NUM_INPUTS = 2
const NUM_HIDDEN = 4
const NUM_OUTPUTS = 2
const NUM_SAMPLES = 4
const LEARNING_RATE = 0.3
const EPOCHS = 10000

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

// Sigmoid and derivative
function sigmoid(x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

function sigmoid_derivative(x) {
    return x * (1.0 - x); // assumes x = sigmoid(x)
}

function generateRandomMatrix(row, col) {
    matrix = [];
    for (let i = 0; i < row; i++) {
        matrix[i] = []
        for (let j = 0; j < col; j++) {
            matrix[i][j] = parseFloat((Math.random() * 2 - 1).toFixed(3));
            // random number between -1 and 1, limited to 3 decimal places
        }
    }
    return matrix;
}

function generateRandomArray(n) {
    arr = []
    for (let i = 0; i < n; i++) {
        arr[i] = parseFloat((Math.random() * 2 - 1).toFixed(3));
    }
    return arr;
}

// Weights and biases
let Wh = generateRandomMatrix(NUM_HIDDEN, NUM_INPUTS);
let Bh = generateRandomArray(NUM_HIDDEN);
let Wo = generateRandomMatrix(NUM_OUTPUTS, NUM_HIDDEN);
let Bo = generateRandomArray(NUM_OUTPUTS);

// Wh = [[5.894844, -5.672763], [6.295306, -6.373247]]
// Bh = [2.835286, -3.422918]
// Wo = [[-9.206968, 9.411332]]
// Bo = [4.384727]

function train() {
    for (let epoch = 0; epoch < EPOCHS; epoch++) {
        let totalError = 0.0;
        // loop over all the samples considering them one batch
        // then apply forward pass for all of them
        for (let s = 0; s < NUM_SAMPLES; s++) {
            // all the arrays i.e., output, error and delta for hidden and output
            // are of same dimensions and i.e. equal to number of nodes in that layer 
            // i.e., NUM_HIDDEN and NUM_OUTPUT resp
            const Ho = new Array(NUM_HIDDEN).fill(0);
            const Oo = new Array(NUM_OUTPUTS).fill(0);

            const Oe = new Array(NUM_OUTPUTS).fill(0);
            const Od = new Array(NUM_OUTPUTS).fill(0);

            const He = new Array(NUM_HIDDEN).fill(0);
            const Hd = new Array(NUM_HIDDEN).fill(0);

            // --- Forward pass ---
            for (let i = 0; i < NUM_HIDDEN; i++) {
                Ho[i] = Bh[i];
                for (let j = 0; j < NUM_INPUTS; j++) {
                    Ho[i] += Wh[i][j] * data.inputs[s][j];
                }
                Ho[i] = sigmoid(Ho[i]);
            }

            for (let i = 0; i < NUM_OUTPUTS; i++) {
                Oo[i] = Bo[i];
                for (let j = 0; j < NUM_HIDDEN; j++) {
                    Oo[i] += Wo[i][j] * Ho[j];
                }
                Oo[i] = sigmoid(Oo[i]);
            }
            // --- Forward pass ---

            // --- Error and loss ---
            for (let i = 0; i < NUM_OUTPUTS; i++) {
                Oe[i] = Oo[i] - data.labels[s][i]; // (a - y), (outputs - labels)
                totalError += Oe[i] * Oe[i];
            }
            // --- Error and loss ---
            // ___ TODO: merge the above and below loop to one ___
            // --- Backpropagation ---
            for (let i = 0; i < NUM_OUTPUTS; i++)
                Od[i] = Oe[i] * sigmoid_derivative(Oo[i]); // this can be merged in the err loop

            for (let i = 0; i < NUM_HIDDEN; i++) {
                for (let j = 0; j < NUM_OUTPUTS; j++)
                    He[i] += Od[j] * Wo[j][i];
                // coorect bcoz here j loop over all the neurons in the output layer
                // and i is for neuron of hidden layer 
                // i.e., Wo[j][i] is the weight between (j)th neuron of output layer 
                // and (i)th neuron of hidden layer
                Hd[i] = He[i] * sigmoid_derivative(Ho[i]);
            }
            // --- Backpropagation ---
            // the logic for this hidden layer can be recurrsively repeated 
            // to implement multiple hidden layers

            // --- Update weights and biases ---
            for (let i = 0; i < NUM_OUTPUTS; i++) {
                for (let j = 0; j < NUM_HIDDEN; j++) {
                    Wo[i][j] -= LEARNING_RATE * Od[i] * Ho[j];
                }
            }
            for (let i = 0; i < NUM_OUTPUTS; i++) {
                Bo[i] -= LEARNING_RATE * Od[i];
            }

            for (let i = 0; i < NUM_HIDDEN; i++) {
                for (let j = 0; j < NUM_INPUTS; j++) {
                    Wh[i][j] -= LEARNING_RATE * Hd[i] * data.inputs[s][j];
                }
            }
            for (let i = 0; i < NUM_HIDDEN; i++) {
                Bh[i] -= LEARNING_RATE * Hd[i];
            }
            // --- Update weights and biases ---
        }

        if (epoch % 1000 == 0)
            console.log(`Epoch ${String(epoch).padStart(4, '0')}, Error: ${(totalError / NUM_SAMPLES).toFixed(3)}`);
    }
    return
}

function main() {
    train();
    console.log("\nTrained XOR Network:");
    for (let s = 0; s < NUM_SAMPLES; s++) {
        const Ho = new Array(NUM_HIDDEN).fill(0);
        const Oo = new Array(NUM_OUTPUTS).fill(0);

        for (let i = 0; i < NUM_HIDDEN; i++) {
            Ho[i] = Bh[i];
            for (let j = 0; j < NUM_INPUTS; j++) {
                Ho[i] += Wh[i][j] * data.inputs[s][j];
            }
            Ho[i] = sigmoid(Ho[i]);
        }

        for (let i = 0; i < NUM_OUTPUTS; i++) {
            Oo[i] = Bo[i];
            for (let j = 0; j < NUM_HIDDEN; j++) {
                Oo[i] += Wo[i][j] * Ho[j];
            }
            Oo[i] = sigmoid(Oo[i]);
        }

        console.log(`Input: ${data.inputs[s][0]} ${data.inputs[s][1]} => Output: ${Oo.map(x => x.toFixed('2'))}`);
    }
    console.log({ Wh, Bh, Wo, Bo });
}

main()