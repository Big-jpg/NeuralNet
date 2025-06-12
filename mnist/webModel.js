
class Layer {
    constructor(numNeurons, numInputs, activation = 'sigmoid') {
        this.numNeurons = numNeurons;
        this.numInputs = numInputs;
        this.activation = activation;

        this.weights = this.generateRandomMatrix(this.numNeurons, this.numInputs);
        this.biases = new Array(this.numNeurons).fill(0);

        this.input = new Array(this.numNeurons).fill(0);
        this.z = new Array(this.numNeurons).fill(0);
        this.output = new Array(this.numNeurons).fill(0);
    }

    generateRandomMatrix(row, col) {
        return Array.from({ length: row }, () =>
            Array.from({ length: col }, () => Math.random() * Math.sqrt(2 / this.numInputs)
            ));
    }

    activate(x) {
        if (this.activation === 'sigmoid') return 1 / (1 + Math.exp(-x));
        if (this.activation === 'relu') return Math.max(0, x);
    }

    activationDerivative(x) {
        if (this.activation === 'sigmoid') return x * (1 - x); // x = sigmoid(x)
        if (this.activation === 'relu') return x > 0 ? 1 : 0;
    }

    softmax(x) {
        const maxX = Math.max(...x);
        const expValues = x.map(value => Math.exp(value - maxX));
        const sumExpValues = expValues.reduce((a, b) => a + b, 0);
        return expValues.map(value => value / sumExpValues);
    }

    forward(input) {
        this.input = [...input];
        for (let i = 0; i < this.numNeurons; i++) {
            this.z[i] = this.biases[i];
            for (let j = 0; j < this.numInputs; j++) {
                this.z[i] += this.weights[i][j] * this.input[j];
            }
        }
        if (this.activation == 'softmax') {
            this.output = this.softmax(this.z);
        } else {
            for (let i = 0; i < this.numNeurons; i++) {
                this.output[i] = this.activate(this.z[i]);
            }
        }
        return this.output;
    }
}

class NeuralNetwork {
    constructor(layers) {
        this.numLayers = layers.length;
        this.layers = [];
        layers.forEach(layer => {
            const newLayer = new Layer(layer.numNeurons, layer.numInputs, layer.activation);
            newLayer.weights = layer.weights;
            newLayer.biases = layer.biases;
            this.layers.push(newLayer);
        });
    }

    forwardPass(inputArr) {
        let dataInput = inputArr;
        for (let i = 0; i < this.numLayers; i++) {
            dataInput = this.layers[i].forward(dataInput);
        }
        return dataInput;
    }

    predict(inputArr) {
        return this.forwardPass(inputArr);
    }

    drawNN(canvasSelector, ignoreInput = false, cwidth = 600, cheight = 400, weightLineWidth = 0.5, nodeRadius = 6) {
        const canvas = document.querySelector(canvasSelector);
        const ctx = canvas.getContext('2d');
        const width = (cwidth - cwidth / 20) || canvas.width - canvas.width / 20;
        const height = (cheight - cheight / 20) || canvas.height - canvas.height / 20;

        ctx.clearRect(0, 0, width, height);

        let layerCount; // no. of nn layers to render
        let layerSizes = [];
        const neuronPositions = [];

        if (ignoreInput) {
            layerCount = this.numLayers; // ignore ip layer
        } else {
            layerCount = this.numLayers + 1; // include ip layer
            layerSizes.push(this.layers[0].numInputs); // first layer numNeurons addded
        }
        this.layers.forEach(layer => layerSizes.push(layer.numNeurons)); // hidd and op layer

        for (let l = 0; l < layerCount; l++) {
            // const numNeurons = Math.min(layerSizes[l], 20);
            // modify this to group 28 neurons together in first layer
            const numNeurons = layerSizes[l];
            const layerX = (width / (layerCount - 1)) * l + 15; //15 to adjust 
            const neuronSpacing = height / (numNeurons + 1);

            const positions = [];
            for (let n = 0; n < numNeurons; n++) {
                const neuronY = neuronSpacing * (n + 1) + 10;
                positions.push({ x: layerX, y: neuronY });
            }
            neuronPositions.push(positions);
        }

        // weight lines - red && green
        ctx.lineWidth = weightLineWidth;
        for (let l = 1; l < layerCount; l++) {
            const prevLayerPositions = neuronPositions[l - 1];
            const currLayerPositions = neuronPositions[l];
            let weights = this.layers[l - 1].weights;
            // because this.layers[0] connects input layer to first hidden layer
            if (ignoreInput) {
                weights = this.layers[l].weights;
                // ignore first hidden layers' weights
            }

            for (let i = 0; i < currLayerPositions.length; i++) {
                for (let j = 0; j < prevLayerPositions.length; j++) {
                    const weight = weights[i][j];

                    ctx.beginPath();
                    ctx.moveTo(prevLayerPositions[j].x, prevLayerPositions[j].y);
                    ctx.lineTo(currLayerPositions[i].x, currLayerPositions[i].y);

                    if (weight >= 0) {
                        ctx.strokeStyle = `rgba(0, 150, 0, ${Math.min(Math.abs(weight), 1)})`;
                    } else {
                        ctx.strokeStyle = `rgba(200, 0, 0, ${Math.min(Math.abs(weight), 1)})`;
                    }
                    ctx.stroke();
                }
            }
        }

        // Draw neurons - blue && yellow
        const neuronRadius = nodeRadius;
        ctx.lineWidth = nodeRadius / 5;
        for (let l = 0; l < layerCount; l++) {
            const layerPositions = neuronPositions[l];

            let isInputLayer = (l === 0);
            let biases = isInputLayer ? null : this.layers[l - 1].biases;

            if (ignoreInput) {
                isInputLayer = false;
                biases = this.layers[l].biases;
            }

            for (let n = 0; n < layerPositions.length; n++) {
                const { x, y } = layerPositions[n];

                if (isInputLayer) {
                    ctx.fillStyle = `rgba(100, 100, 255, 0.8)`; // Input layer neurons
                } else {
                    const bias = biases[n];
                    ctx.fillStyle = bias >= 0
                        ? `rgba(0, 150, 255, 0.9)`
                        : `rgba(255, 150, 0, 0.8)`;
                }

                ctx.beginPath();
                ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#333';
                ctx.stroke();
            }
        }
    }
}