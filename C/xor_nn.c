#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_INPUTS 2
#define NUM_HIDDEN 2
#define NUM_OUTPUTS 1
#define NUM_SAMPLES 4
#define LEARNING_RATE 0.5
#define EPOCHS 10000

// XOR inputs and expected outputs
float inputs[NUM_SAMPLES][NUM_INPUTS] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}};

float expected[NUM_SAMPLES][NUM_OUTPUTS] = {
    {0},
    {1},
    {1},
    {0}};

// Sigmoid and derivative
float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x)
{
    return x * (1.0f - x); // assumes x = sigmoid(x)
}

int main()
{
    srand(time(NULL));

    // Weights and biases
    float hidden_weights[NUM_INPUTS][NUM_HIDDEN];
    float hidden_bias[NUM_HIDDEN];
    float output_weights[NUM_HIDDEN][NUM_OUTPUTS];
    float output_bias[NUM_OUTPUTS];

    // Initialize weights and biases
    for (int i = 0; i < NUM_INPUTS; i++)
        for (int j = 0; j < NUM_HIDDEN; j++)
            hidden_weights[i][j] = ((float)rand() / RAND_MAX) * 2 - 1;

    for (int i = 0; i < NUM_HIDDEN; i++)
        hidden_bias[i] = 0.0f;

    for (int i = 0; i < NUM_HIDDEN; i++)
        for (int j = 0; j < NUM_OUTPUTS; j++)
            output_weights[i][j] = ((float)rand() / RAND_MAX) * 2 - 1;

    for (int i = 0; i < NUM_OUTPUTS; i++)
        output_bias[i] = 0.0f;

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        float total_error = 0.0f;

        for (int s = 0; s < NUM_SAMPLES; s++)
        {
            float hidden[NUM_HIDDEN];
            float output[NUM_OUTPUTS];

            // --- Forward pass ---
            for (int i = 0; i < NUM_HIDDEN; i++)
            {
                hidden[i] = hidden_bias[i];
                for (int j = 0; j < NUM_INPUTS; j++)
                    hidden[i] += inputs[s][j] * hidden_weights[j][i];
                hidden[i] = sigmoid(hidden[i]);
            }

            for (int i = 0; i < NUM_OUTPUTS; i++)
            {
                output[i] = output_bias[i];
                for (int j = 0; j < NUM_HIDDEN; j++)
                    output[i] += hidden[j] * output_weights[j][i];
                output[i] = sigmoid(output[i]);
            }

            // --- Error and loss ---
            float output_error[NUM_OUTPUTS];
            for (int i = 0; i < NUM_OUTPUTS; i++)
            {
                output_error[i] = expected[s][i] - output[i];
                total_error += output_error[i] * output_error[i];
            }

            // --- Backpropagation ---
            float output_delta[NUM_OUTPUTS];
            for (int i = 0; i < NUM_OUTPUTS; i++)
                output_delta[i] = output_error[i] * sigmoid_derivative(output[i]);

            float hidden_error[NUM_HIDDEN] = {0};
            float hidden_delta[NUM_HIDDEN];

            for (int i = 0; i < NUM_HIDDEN; i++)
            {
                for (int j = 0; j < NUM_OUTPUTS; j++)
                    hidden_error[i] += output_delta[j] * output_weights[i][j];
                hidden_delta[i] = hidden_error[i] * sigmoid_derivative(hidden[i]);
            }

            // --- Update weights and biases ---
            for (int i = 0; i < NUM_HIDDEN; i++)
                for (int j = 0; j < NUM_OUTPUTS; j++)
                    output_weights[i][j] += LEARNING_RATE * output_delta[j] * hidden[i];

            for (int i = 0; i < NUM_OUTPUTS; i++)
                output_bias[i] += LEARNING_RATE * output_delta[i];

            for (int i = 0; i < NUM_INPUTS; i++)
                for (int j = 0; j < NUM_HIDDEN; j++)
                    hidden_weights[i][j] += LEARNING_RATE * hidden_delta[j] * inputs[s][i];

            for (int i = 0; i < NUM_HIDDEN; i++)
                hidden_bias[i] += LEARNING_RATE * hidden_delta[i];
        }

        if (epoch % 1000 == 0)
            printf("Epoch %d, Error: %f\n", epoch, total_error / NUM_SAMPLES);
    }

    // Final output
    printf("\nWEIGHTS 1\n");
    for (int i = 0; i < NUM_HIDDEN; i++)
    {
        for (int j = 0; j < NUM_INPUTS; j++)
        {
            printf(" %f ,", hidden_weights[j][i]);
        }
    }
    printf("\nBIASES 1\n");
    for (int i = 0; i < NUM_HIDDEN; i++)
    {
        printf(" %f ,", hidden_bias[i]);
    }
    printf("\nWEIGHTS 2\n");
    for (int i = 0; i < NUM_OUTPUTS; i++)
    {
        for (int j = 0; j < NUM_HIDDEN; j++)
        {
            printf(" %f ,", output_weights[j][i]);
        }
    }
    printf("\nBIASES 2\n");
    for (int i = 0; i < NUM_OUTPUTS; i++)
    {
        printf(" %f ,", output_bias[i]);
    }

    printf("\nTrained XOR Network:\n");
    for (int s = 0; s < NUM_SAMPLES; s++)
    {
        float hidden[NUM_HIDDEN];
        float output[NUM_OUTPUTS];

        for (int i = 0; i < NUM_HIDDEN; i++)
        {
            hidden[i] = hidden_bias[i];
            for (int j = 0; j < NUM_INPUTS; j++)
                hidden[i] += inputs[s][j] * hidden_weights[j][i];
            hidden[i] = sigmoid(hidden[i]);
        }

        for (int i = 0; i < NUM_OUTPUTS; i++)
        {
            output[i] = output_bias[i];
            for (int j = 0; j < NUM_HIDDEN; j++)
                output[i] += hidden[j] * output_weights[j][i];
            output[i] = sigmoid(output[i]);
        }

        printf("Input: %d %d => Output: %.4f\n",
               (int)inputs[s][0], (int)inputs[s][1], output[0]);
    }

    return 0;
}
