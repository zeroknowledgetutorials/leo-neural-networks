function main(w00: u32, b00: u32, w01: u32, b01: u32, w100: u32, w101: u32, b10: u32, w110: u32, w111: u32, b11: u32, w200: u32, b20: u32, w210: u32, b21: u32, input0: u32, input1: u32) -> [u32; 1] {
let neuron00: u32 = w00 * input0 / 2 + b00;
let neuron01: u32 = w01 * input1 / 2 + b01;
let neuron10: u32 = rectified_linear_activation(neuron00 * w100 / 2 + neuron01 * w110 / 2 + b10);
let neuron11: u32 = rectified_linear_activation(neuron00 * w101 / 2 + neuron01 * w111 / 2 + b11);
let neuron20: u32 = rectified_linear_activation(neuron10 * w200 / 2 + neuron11 * w210 / 2 + b20);
return [neuron20];}

function rectified_linear_activation(x: u32) -> u32 {
let result: u32 = 0;
if x > 0 {
result = x;
}
return result;
}