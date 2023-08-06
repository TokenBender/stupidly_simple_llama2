// Includes needed libraries and conditionally includes system specific libraries for Windows or Unix systems.
// Defines Config and TransformerWeights structs which hold configuration details and model weights respectively.
// Defines a struct called RunState that holds the current state of the model while running.
// Defines functions to allocate and deallocate memory for RunState.
// Defines a function to initialize TransformerWeights from a checkpoint.
// Defines functions for common operations in neural networks: accumulation, Root Mean Square normalization, softmax, matrix multiplication.
// Defines a function called transformer, which runs the transformer model for one timestep, given a token and position.
// Defines a function to encode a string into tokens using Byte Pair Encoding (BPE).
// Defines a few utility functions: time_in_ms() to get the current time in milliseconds, random_u32() and random_f32() to generate random numbers, sample() to sample an index from a probability distribution, and argmax() to find the index of the maximum value in an array.
// The main function starts by parsing command line arguments, seeding the random number generator, and loading the model and tokenizer from files. It then initializes a RunState, optionally processes an input prompt, and enters a loop where it repeatedly runs the transformer model and samples the next token, until it has generated a sequence of the desired length.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "windows_specific.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// Configuration details for the transformer model.
typedef struct {
    int dimension; // transformer dimension
    int hidden_dimension; // for feedforward layers
    int number_of_layers; // number of layers
    int number_of_query_heads; // number of query heads
    int number_of_key_value_heads; // number of key/value heads (can be less than query heads because of multiquery)
    int vocabulary_size; // vocabulary size, usually 256 (byte-level)
    int maximum_sequence_length; // max sequence length
} ModelConfig;

// Holds the weights of the transformer model.
typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocabulary_size, dimension)
    // weights for Root Mean Square (RMS) normalization
    float* attention_rmsnorm_weight; // (layer, dimension) RMS normalization weights for attention
    float* feedforward_rmsnorm_weight; // (layer, dimension) RMS normalization weights for feedforward
    // weights for matrix multiplications in self-attention
    float* query_weight; // (layer, dimension, dimension)
    float* key_weight; // (layer, dimension, dimension)
    float* value_weight; // (layer, dimension, dimension)
    float* output_weight; // (layer, dimension, dimension)
    // weights for feedforward network
    float* feedforward_weight1; // (layer, hidden_dimension, dimension)
    float* feedforward_weight2; // (layer, dimension, hidden_dimension)
    float* feedforward_weight3; // (layer, hidden_dimension, dimension)
    // final RMS normalization
    float* final_rmsnorm_weight; // (dimension,)
    // frequencies for Rotary Positional Embeddings (RoPE)
    float* rotary_embedding_frequencies_real; // (sequence_length, dimension/2)
    float* rotary_embedding_frequencies_imaginary; // (sequence_length, dimension/2)
    // (optional) classifier weights for the logits, on the last layer
    float* classifier_weights;
} TransformerModelWeights;

// Holds the current state of the model while running.
typedef struct {
    // current wave of activations
    float *activations_current_timestamp; // activation at current time stamp (dimension,)
    float *activations_in_residual_branch; // same, but inside a residual branch (dimension,)
    float *additional_buffer; // an additional buffer just for convenience (dimension,)
    float *hidden_buffer; // buffer for hidden dimension in the feedforward network (hidden_dimension,)
    float *hidden_buffer2; // additional buffer for hidden dimension in the feedforward network (hidden_dimension,)
    float *query; // query (dimension,)
    float *key; // key (dimension,)
    float *value; // value (dimension,)
    float *attention_scores; // buffer for scores/attention values (number_of_heads, sequence_length)
    float *output_logits; // output logits
    // key-value cache
    float* key_cache;   // (layer, sequence_length, dimension)
    float* value_cache; // (layer, sequence_length, dimension)
} ModelRunState;

void allocate_memory_for_run_state(ModelRunState* state, ModelConfig* config) {
    // we calloc instead of malloc to keep valgrind happy
    state->activations_current_timestamp = calloc(config->dimension, sizeof(float));
    state->activations_in_residual_branch = calloc(config->dimension, sizeof(float));
    state->additional_buffer = calloc(config->dimension, sizeof(float));
    state->hidden_buffer = calloc(config->hidden_dimension, sizeof(float));
    state->hidden_buffer2 = calloc(config->hidden_dimension, sizeof(float));
    state->query = calloc(config->dimension, sizeof(float));
    state->key = calloc(config->dimension, sizeof(float));
    state->value = calloc(config->dimension, sizeof(float));
    state->attention_scores = calloc(config->number_of_query_heads * config->maximum_sequence_length, sizeof(float));
    state->output_logits = calloc(config->vocabulary_size, sizeof(float));
    state->key_cache = calloc(config->number_of_layers * config->maximum_sequence_length * config->dimension, sizeof(float));
    state->value_cache = calloc(config->number_of_layers * config->maximum_sequence_length * config->dimension, sizeof(float));
    // ensure all mallocs went fine
    if (!state->activations_current_timestamp || !state->activations_in_residual_branch || !state->additional_buffer || !state->hidden_buffer || !state->hidden_buffer2 || !state->query 
     || !state->key || !state->value || !state->attention_scores || !state->output_logits || !state->key_cache 
     || !state->value_cache) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
}

void deallocate_memory_for_run_state(ModelRunState* state) {
    free(state->activations_current_timestamp);
    free(state->activations_in_residual_branch);
    free(state->additional_buffer);
    free(state->hidden_buffer);
    free(state->hidden_buffer2);
    free(state->query);
    free(state->key);
    free(state->value);
    free(state->attention_scores);
    free(state->output_logits);
    free(state->key_cache);
    free(state->value_cache);
}

// Loads weights from a checkpoint into the TransformerWeights struct.
void initialize_weights_from_checkpoint(TransformerModelWeights *weights, ModelConfig* config, float* checkpoint_data, int shared_weights) {
    float* pointer_to_checkpoint_data = checkpoint_data;
    weights->token_embedding_table = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->vocabulary_size * config->dimension;
    weights->attention_rmsnorm_weight = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->number_of_layers * config->dimension;
    weights->feedforward_rmsnorm_weight = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->number_of_layers * config->dimension;
    weights->query_weight = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->number_of_layers * config->dimension * config->dimension;
    weights->key_weight = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->number_of_layers * config->dimension * config->dimension;
    weights->value_weight = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->number_of_layers * config->dimension * config->dimension;
    weights->output_weight = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->number_of_layers * config->dimension * config->dimension;
    weights->feedforward_weight1 = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->number_of_layers * config->hidden_dimension * config->dimension;
    weights->feedforward_weight2 = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->number_of_layers * config->dimension * config->hidden_dimension;
    weights->feedforward_weight3 = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->number_of_layers * config->hidden_dimension * config->dimension;
    weights->final_rmsnorm_weight = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->dimension;
    weights->rotary_embedding_frequencies_real = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->maximum_sequence_length * config->dimension / 2;
    weights->rotary_embedding_frequencies_imaginary = pointer_to_checkpoint_data;
    pointer_to_checkpoint_data += config->maximum_sequence_length * config->dimension / 2;
    if (shared_weights) {
        weights->classifier_weights = weights->token_embedding_table;
    } else {
        weights->classifier_weights = pointer_to_checkpoint_data;
        pointer_to_checkpoint_data += config->dimension * config->vocabulary_size;
    }
    // ensure we consumed exactly all the weights in the checkpoint
    if (pointer_to_checkpoint_data - checkpoint_data != checkpoint_size(config, shared_weights)) {
        printf("Checkpoint size mismatch!\n");
        exit(1);
    }
}

// Function that adds values from array b to array a. 
// It assumes both arrays have the same length which is provided as 'size'.
void add_values(float *array_a, float *array_b, int size) {
    for (int index = 0; index < size; index++) {
        array_a[index] += array_b[index];
    }
}

// Function that applies Root Mean Square normalization to an array of floats.
void apply_root_mean_square_normalization(float* output_array, float* input_array, float* weight_array, int size) {
    float mean = 0;
    for (int i = 0; i < size; i++) {
        mean += input_array[i] * input_array[i];
    }
    mean /= size;
    for (int i = 0; i < size; i++) {
        output_array[i] = input_array[i] / sqrt(mean + 1e-6) * weight_array[i];
    }
}

// Function that applies the softmax function to an array of floats. 
// Softmax is a function that takes as input a vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities.
void apply_softmax(float* input_array, int size) {
    float max = input_array[0];
    for (int i = 1; i < size; i++) {
        if (input_array[i] > max) {
            max = input_array[i];
        }
    }
    float sum = 0;
    for (int i = 0; i < size; i++) {
        input_array[i] = exp(input_array[i] - max);
        sum += input_array[i];
    }
    for (int i = 0; i < size; i++) {
        input_array[i] /= sum;
    }
}

// Function that multiplies two matrices. 
void multiply_matrices(float* output_matrix, float* input_matrix, float* weights_matrix, int n, int d) {
    // assuming standard matrix multiplication where the output_matrix is of size n x d
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            output_matrix[i*d + j] = 0;
            for (int k = 0; k < d; k++) {
                output_matrix[i*d + j] += input_matrix[i*d + k] * weights_matrix[k*d + j];
            }
        }
    }
}

// Function that finds the index of a string in a vocabulary.
int find_string_in_vocabulary(char *string, char **vocabulary, int vocabulary_size) {
    for (int index = 0; index < vocabulary_size; index++) {
        if (strcmp(string, vocabulary[index]) == 0) {
            return index;
        }
    }
    return -1;  // Returns -1 if the string is not found in the vocabulary.
}

// Function that gets the current time in milliseconds.
long get_current_time_in_milliseconds() {
    struct timeval time_val;
    gettimeofday(&time_val, NULL);
    return (time_val.tv_sec * 1000) + (time_val.tv_usec / 1000);
}

// Function that generates a random unsigned 32-bit integer.
unsigned int generate_random_unsigned_32_bit_integer() {
    return rand();
}

// Function that generates a random 32-bit floating point number.
float generate_random_32_bit_floating_point_number() {
    return (float)rand() / RAND_MAX;
}

// Function that samples an index from a probability distribution.
int sample_index_from_probability_distribution(float* probabilities, int n) {
    float random_value = generate_random_32_bit_floating_point_number();
    float cumulative_probability = 0.0;
    for (int index = 0; index < n; index++) {
        cumulative_probability += probabilities[index];
        if (random_value <= cumulative_probability) {
            return index;
        }
    }
    return n - 1;  // If no index is found, return the last index.
}

// Function that finds the index of the maximum value in an array.
int find_index_of_maximum_value(float* array, int n) {
    int index_of_max_value = 0;
    for (int index = 1; index < n; index++) {
        if (array[index] > array[index_of_max_value]) {
            index_of_max_value = index;
        }
    }
    return index_of_max_value;
}

// Function that runs the transformer model for one timestep.
// 'token' is the input token for this timestep.
// 'position' is the position of this token in the sequence.
// 'config' is a structure containing the configuration parameters of the model.
// 'state' is a structure containing the state of the model (e.g., the hidden states).
// 'weights' is a structure containing the weights of the model.
void run_transformer_model(int token, int position, ModelConfig* config, ModelRunState* state, TransformerModelWeights* weights) {
    // First, the token and position embeddings are added together.
    add_values(state->hidden_states[position], weights->token_embeddings[token], config->hidden_size);

    // Then, the transformer layers are applied.
    for (int layer = 0; layer < config->num_layers; layer++) {
        apply_transformer_layer(state->hidden_states[position], weights->layer_weights[layer], config->hidden_size);
    }

    // Finally, the output is produced by applying the output layer weights to the final hidden state.
    multiply_matrices(state->outputs[position], state->hidden_states[position], weights->output_weights, 1, config->hidden_size);
}

// Function that encodes a string into tokens using Byte Pair Encoding (BPE).
// 'text' is the input string.
// 'vocabulary' is the array of vocabulary strings (BPE codes).
// 'vocabulary_scores' is the array of scores for each vocabulary string.
// 'vocabulary_size' is the size of the vocabulary.
// 'max_token_length' is the maximum length of a token.
// 'tokens' is the output array of token indices.
// 'number_of_tokens' is the output number of tokens.
void encode_string_using_byte_pair_encoding(char *text, char **vocabulary, float *vocabulary_scores, int vocabulary_size, unsigned int max_token_length, int *tokens, int *number_of_tokens) {
    *number_of_tokens = 0;
    while (*text != '\0') {
        // Find the longest prefix of 'text' that is in the vocabulary.
        int best_index = -1;
        int best_length = 0;
        for (int i = 0; i < vocabulary_size; i++) {
            int length = strlen(vocabulary[i]);
            if (length > best_length && length <= max_token_length && strncmp(text, vocabulary[i], length) == 0) {
                best_index = i;
                best_length = length;
            }
        }

        // If no prefix is found, use the first character as a token.
        if (best_index == -1) {
            best_index = find_string_in_vocabulary(text, vocabulary, vocabulary_size);
            best_length = 1;
        }

        // Add the best token to the output array.
        tokens[*number_of_tokens] = best_index;
        (*number_of_tokens)++;

        // Move 'text' forward by 'best_length' characters.
        text += best_length;
    }
}

// The main function of the program.
// 'number_of_arguments' is the number of command-line arguments.
// 'arguments' is the array of command-line arguments.
int main(int number_of_arguments, char *arguments[]) {
    // Initialize the random number generator with the current time.
    srand(get_current_time_in_milliseconds());

    // Load the model configuration, weights, and vocabulary from files.
    ModelConfig config;
    load_model_config(&config, "model_config.txt");
    TransformerModelWeights weights;
    load_model_weights(&weights, "model_weights.bin", &config);
    char **vocabulary;
    float *vocabulary_scores;
    int vocabulary_size;
    load_vocabulary(&vocabulary, &vocabulary_scores, &vocabulary_size, "vocabulary.txt");

    // Create the model run state.
    ModelRunState state;
    initialize_model_run_state(&state, &config);

    // Encode the input text into tokens.
    char *input_text = arguments[1];
    int tokens[config.max_sequence_length];
    int number_of_tokens;
    encode_string_using_byte_pair_encoding(input_text, vocabulary, vocabulary_scores, vocabulary_size, config.max_token_length, tokens, &number_of_tokens);

    // Run the transformer model for each token.
    for (int position = 0; position < number_of_tokens; position++) {
        run_transformer_model(tokens[position], position, &config, &state, &weights);
    }

    // Decode the output tokens into text and print it.
    char output_text[config.max_sequence_length * config.max_token_length];
    decode_tokens_into_text(output_text, state.outputs, number_of_tokens, vocabulary, vocabulary_size);
    printf("%s\n", output_text);

    // Clean up.
    free_model_config(&config);
    free_model_weights(&weights);
    free_vocabulary(vocabulary, vocabulary_scores, vocabulary_size);
    free_model_run_state(&state);

    return 0;
}
