// The 'neuralnetwork-initial' main function.
function main(x0: u32, x1: u32, w00: u32, w01: u32, w100: u32, w101: u32, w110: u32, w111: u32, w20: u32, w21: u32, b00: u32, b01: u32, b10: u32, b11: u32, b2: u32) -> u32 {

let multiplication_correction: u32 = 100; // 10 to the power of 2, the number of decimal points

let neuron00: u32 = w00 * x0 / multiplication_correction + b00;
let neuron01: u32 = w01 * x1 / multiplication_correction + b01;

let neuron10: u32 = rectified_linear_activation(w100 * neuron00 / multiplication_correction + w110 * neuron01 / multiplication_correction + b10);
let neuron11: u32 = rectified_linear_activation(w101 * neuron00 / multiplication_correction + w111 * neuron01 / multiplication_correction + b11);

let neuron20: u32 = rectified_linear_activation(w20 * neuron10 / multiplication_correction + w21 * neuron11 / multiplication_correction + b2);
return neuron20;}

function rectified_linear_activation(x: u32) -> u32 {

    let result: u32 = 0;

    if x > 0 {
        result = x;
    }

    return result;
}