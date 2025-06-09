# VNN.js  

## Current  
- JS class structure  
- seperate forward, backward and update functions  
- SGD - stochastic GD
- RELU for hidden layers  
- support for deepNN
    - using another object for each layer(relu/sigmoid)
    - and using an array of layers and recurrsively updating w&b
    - final: **separated Layers class from NN**
- cross entropy loss for classification

## Improvements  
- mini batch GD with Deep NN  
- softmax for last layer(only when NUM_OUTPUTS > 1), else use sigmoid