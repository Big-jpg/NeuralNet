const fs = require('fs');
const path = require('path');

class NeuralNetwork {
    constructor(NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS, LEARNING_RATE, EPOCHS) {
        this.NUM_INPUTS = NUM_INPUTS;
        this.NUM_HIDDEN = NUM_HIDDEN;
        this.NUM_OUTPUTS = NUM_OUTPUTS;
        this.LEARNING_RATE = LEARNING_RATE;
        this.EPOCHS = EPOCHS;

        this.Wh = this.generateRandomMatrix(this.NUM_HIDDEN, this.NUM_INPUTS);
        this.Bh = this.generateRandomArray(this.NUM_HIDDEN);
        this.Wo = this.generateRandomMatrix(this.NUM_OUTPUTS, this.NUM_HIDDEN);
        this.Bo = this.generateRandomArray(this.NUM_OUTPUTS);

        this.totalError = 0.0;
    }

    sigmoid(x) {
        return 1.0 / (1.0 + Math.exp(-x));
        // return Math.max(0, x);
    }

    sigmoid_derivative(x) {
        return x * (1.0 - x); // assumes x = sigmoid(x)
        // return x > 0 ? 1 : 0; 
    }

    generateRandomMatrix(row, col) {
        const matrix = [];
        for (let i = 0; i < row; i++) {
            matrix[i] = []
            for (let j = 0; j < col; j++) {
                matrix[i][j] = parseFloat(Math.random());
            }
        }
        return matrix;
    }

    generateRandomArray(n) {
        const arr = []
        for (let i = 0; i < n; i++) {
            arr[i] = parseFloat(Math.random());
        }
        return arr;
    }

    forwardPass(inputArr, Ho, Oo) {
        // --- Forward pass ---
        for (let i = 0; i < this.NUM_HIDDEN; i++) {
            Ho[i] = this.Bh[i];
            for (let j = 0; j < this.NUM_INPUTS; j++) {
                Ho[i] += this.Wh[i][j] * inputArr[j];
            }
            Ho[i] = this.sigmoid(Ho[i]);
        }

        for (let i = 0; i < this.NUM_OUTPUTS; i++) {
            Oo[i] = this.Bo[i];
            for (let j = 0; j < this.NUM_HIDDEN; j++) {
                Oo[i] += this.Wo[i][j] * Ho[j];
            }
            Oo[i] = this.sigmoid(Oo[i]);
        }
        // --- Forward pass ---
    }

    backwardPass(labelArr, Ho, He, Hd, Oo, Oe, Od) {
        // --- Error and loss WITH Backpropagation ---
        for (let i = 0; i < this.NUM_OUTPUTS; i++) {
            Oe[i] = Oo[i] - labelArr[i];
            Od[i] = Oe[i] * this.sigmoid_derivative(Oo[i]);
            this.totalError += Oe[i] * Oe[i];
        }
        // console.log("[debug]: totalError:", this.totalError);
        // Oe and Od found

        for (let i = 0; i < this.NUM_HIDDEN; i++) {
            for (let j = 0; j < this.NUM_OUTPUTS; j++)
                He[i] += Od[j] * this.Wo[j][i];
            Hd[i] = He[i] * this.sigmoid_derivative(Ho[i]);
        }
        // He and Hd found
        // --- Backpropagation ---
    }

    // needs to be changes to accomodate average gradient updation
    updateWB(inputArr, Ho, Hd, Od) {
        // --- Update weights and biases ---
        for (let i = 0; i < this.NUM_OUTPUTS; i++) {
            for (let j = 0; j < this.NUM_HIDDEN; j++) {
                this.Wo[i][j] -= this.LEARNING_RATE * Od[i] * Ho[j];
            }
        }
        for (let i = 0; i < this.NUM_OUTPUTS; i++) {
            this.Bo[i] -= this.LEARNING_RATE * Od[i];
        }

        for (let i = 0; i < this.NUM_HIDDEN; i++) {
            for (let j = 0; j < this.NUM_INPUTS; j++) {
                this.Wh[i][j] -= this.LEARNING_RATE * Hd[i] * inputArr[j];
            }
        }
        for (let i = 0; i < this.NUM_HIDDEN; i++) {
            this.Bh[i] -= this.LEARNING_RATE * Hd[i];
        }
        // --- Update weights and biases ---
    }

    train(TRAINING_DATA, BATCH_SIZE) {
        const NUM_SAMPLES = Math.min(TRAINING_DATA.inputs.length, TRAINING_DATA.labels.length);
        const NUM_BATCHES = Math.ceil(NUM_SAMPLES / BATCH_SIZE);
        for (let epoch = 0; epoch < this.EPOCHS; epoch++) {
            // // --- randomize inputs for batch making ---
            // function swap(arr1, arr2, i, rand) {
            //     [arr1[i], arr1[rand]] = [arr1[rand], arr1[i]];
            //     [arr2[i], arr2[rand]] = [arr2[rand], arr2[i]];
            // }

            // for (let i = 0; i < NUM_SAMPLES; i++) {
            //     const rand = parseInt(Math.random() * NUM_SAMPLES) % NUM_SAMPLES;
            //     swap(TRAINING_DATA.inputs, TRAINING_DATA.labels, i, rand);
            // }
            // // --- randomize inputs for batch making ---

            // --- create batches ---
            let batchStart = 0, batchEnd = BATCH_SIZE;
            batchEnd = Math.min(batchEnd, NUM_SAMPLES);
            // --- create batches ---

            for (let b = 0; b < NUM_BATCHES; b++) {
                // --- singular batch ---
                for (let s = batchStart; s < batchEnd; s++) {
                    // --- singular input ---
                    const Ho = new Array(this.NUM_HIDDEN).fill(0);
                    const He = new Array(this.NUM_HIDDEN).fill(0);
                    const Hd = new Array(this.NUM_HIDDEN).fill(0);

                    const Oo = new Array(this.NUM_OUTPUTS).fill(0);
                    const Oe = new Array(this.NUM_OUTPUTS).fill(0);
                    const Od = new Array(this.NUM_OUTPUTS).fill(0);

                    this.forwardPass(TRAINING_DATA.inputs[s], Ho, Oo);
                    this.backwardPass(TRAINING_DATA.labels[s], Ho, He, Hd, Oo, Oe, Od);
                    this.updateWB(TRAINING_DATA.inputs[s], Ho, Hd, Od);
                    // --- singular input ---

                    // currently each and every data point is updating weights immediately
                    // implement batches and update gradients' average
                }
                // --- singular batch ---
                batchStart += BATCH_SIZE;
                batchEnd += BATCH_SIZE;
                batchEnd = Math.min(batchEnd, NUM_SAMPLES);
            }

            if (epoch % 1000 == 0) {
                console.log(`| Epoch ${String(epoch).padStart(4, '0')} | Error: ${(this.totalError / NUM_SAMPLES).toFixed(3)} |`);
            }
            this.totalError = 0.0;
        }
    }

    predict(dataArr) {
        // dataArr: array having input neurons' values -> [n]
        const Ho = []
        const Oo = []
        this.forwardPass(dataArr, Ho, Oo);
        // return this.softmax(Oo);
        return Oo;
    }

    softmax(x) {
        const maxX = Math.max(...x);
        const expValues = x.map(value => Math.exp(value - maxX));
        const sumExpValues = expValues.reduce((acc, curr) => acc + curr, 0);
        return expValues.map(value => value / sumExpValues);
    }

    dumpWB() {
        const filePath = path.join(__dirname, 'WeightsBiases.json');
        fs.writeFile(filePath, JSON.stringify({
            WeightsHidden: this.Wh,
            BiasesHidden: this.Bh,
            WeightsOutput: this.Wo,
            BiasesOutput: this.Bo
        }), (err) => {
            if (err) throw err;
            console.log('The file has been saved!');
        });
        // console.log({
        //     WeightsHidden: this.Wh,
        //     BiasesHidden: this.Bh,
        //     WeightsOutput: this.Wo,
        //     BiasesOutput: this.Bo
        // });
    }

    async loadWB() {
        const filePath = path.join(__dirname, 'WeightsBiases.json');

        try {
            // Use fs.promises.readFile to read the file asynchronously
            const data = await fs.promises.readFile(filePath, 'utf8');

            const obj = JSON.parse(data);

            // Load the weights and biases
            this.Wh = obj.WeightsHidden;
            this.Bh = obj.BiasesHidden;
            this.Wo = obj.WeightsOutput;
            this.Bo = obj.BiasesOutput;

            // Now that data is loaded, log the result
            // console.log(obj);  // Will properly log after the file is read

        } catch (err) {
            console.error("Error loading the file:", err);
        }
    }
}

module.exports = NeuralNetwork;