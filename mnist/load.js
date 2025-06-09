const fs = require('fs');

class MnistDataloader {
    constructor(trainingImagesFilePath, trainingLabelsFilePath, testImagesFilePath, testLabelsFilePath) {
        this.trainingImagesFilePath = trainingImagesFilePath;
        this.trainingLabelsFilePath = trainingLabelsFilePath;
        this.testImagesFilePath = testImagesFilePath;
        this.testLabelsFilePath = testLabelsFilePath;
    }

    readImagesLabels(imagesFilePath, labelsFilePath) {
        let labels = [];
        let images = [];

        // Read labels
        const labelData = fs.readFileSync(labelsFilePath);
        const labelMagic = labelData.readUInt32BE(0);
        if (labelMagic !== 2049) {
            throw new Error('Magic number mismatch, expected 2049');
        }

        const labelCount = labelData.readUInt32BE(4);
        labels = Array.from(labelData.slice(8));  // Starting from the 9th byte for labels

        // Read image data
        const imageData = fs.readFileSync(imagesFilePath);
        const imageMagic = imageData.readUInt32BE(0);
        if (imageMagic !== 2051) {
            throw new Error('Magic number mismatch, expected 2051');
        }

        const imageCount = imageData.readUInt32BE(4);
        const rows = imageData.readUInt32BE(8);
        const cols = imageData.readUInt32BE(12);

        let imageOffset = 16; // Image data starts from byte 16
        for (let i = 0; i < imageCount; i++) {
            const img = Array.from(imageData.slice(imageOffset, imageOffset + rows * cols));
            images.push(img);
            imageOffset += rows * cols;
        }

        // one hot encoding
        labels = labels.map(lbl => {
            const arr = new Array(10).fill(0);
            arr[lbl] = 1
            return arr;
        })

        return { images, labels };
    }

    loadData() {
        const train = this.readImagesLabels(this.trainingImagesFilePath, this.trainingLabelsFilePath);
        const test = this.readImagesLabels(this.testImagesFilePath, this.testLabelsFilePath);
        return {
            // train: { images: train.images, labels: train.labels },
            // test: { images: test.images, labels: test.labels }
            train: { inputs: train.images, labels: train.labels },
            test: { inputs: test.images, labels: test.labels }
        };
    }
}

// LOADING AND EXPORTING
const trainingImagesFilePath = './data/train-images.idx3-ubyte';
const trainingLabelsFilePath = './data/train-labels.idx1-ubyte';
const testImagesFilePath = './data/t10k-images.idx3-ubyte';
const testLabelsFilePath = './data/t10k-labels.idx1-ubyte';

const mnistDataLoader = new MnistDataloader(
    trainingImagesFilePath,
    trainingLabelsFilePath,
    testImagesFilePath,
    testLabelsFilePath
);

// const { train, test } = mnistDataLoader.loadData();
// module.exports = { train, test };

module.exports = { train, test } = mnistDataLoader.loadData();

// console.log(train)
// {
//     inputs: [],
//     labels: []
// }

// fs.writeFile("./imageData.json", JSON.stringify({
//     input: train.images[0], label: train.labels[0]
// }), (err) => {
//     if (err) throw new Error(err);
//     else console.log("image saved");
// })


// // To show images (Node.js doesn't support direct image rendering, so you would need a web server for that)
// // Here’s a function that you can use to visualize the data on an HTML page
// function showImages(images, labels) {
//     const canvas = document.createElement('canvas');
//     const ctx = canvas.getContext('2d');
//     canvas.width = 280;
//     canvas.height = 280;

//     // Render images
//     images.forEach((imageData, index) => {
//         let imgData = ctx.createImageData(28, 28);
//         let data = imgData.data;

//         // Map image data to RGBA for rendering
//         imageData.forEach((pixel, i) => {
//             const baseIdx = i * 4;
//             data[baseIdx] = pixel;  // R
//             data[baseIdx + 1] = pixel;  // G
//             data[baseIdx + 2] = pixel;  // B
//             data[baseIdx + 3] = 255;  // A
//         });

//         ctx.putImageData(imgData, (index % 5) * 60, Math.floor(index / 5) * 60);
//     });
//     // document.body.appendChild(canvas);
//     console.log(canvas);
// }

// // Example usage: Show first 10 training images
// // showImages(train.images.slice(0, 10), train.labels.slice(0, 10));