#include "neural.h"
neuron::neuron(){}
neuron::neuron(size_t input): weights(input) 
{
	weights.random_init();
	weights.requires_grad();
	bias.random_init();
	bias.requires_grad();
}
void neuron::output(value_array& input, activations act)
{
	weights_input = std::move(input* weights);  
	sum = std::move(weights_input.sum());
	sumwbias = std::move(sum + bias); 
	 if(act == TANH){
	 	out = std::move(sumwbias.tanh()); 
	 } else if(act == ELU){
	 	out = std::move(sumwbias.elu());
	 } else{
	 return;
	 }
}
value& neuron::return_output() 
{
	return out;
}
mlp::mlp(){}
mlp::mlp(std::initializer_list<int> temp):layer_neurons(temp) {}
