# VNN.js  

## Current  
- JS class structure  
- seperate forward, backward and update functions  
- SGD - stochastic GD

## Improvements  
- mini batch GD  
- RELU for hidden layers  
- support for deepNN
    - either using another object for each layer(relu/sigmoid)
    - or using an array of layers and recurrsively updating w&b
- softmax for last layer(only when NUM_OUTPUTS > 1), else use sigmoid