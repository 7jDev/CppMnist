#include "neural.h"
neuron::neuron(){}
neuron::neuron(size_t input, activations act): weights(input), neuron_activation(act) 
{
	weights.random_init();
	weights.requires_grad();
	bias.random_init();
	bias.requires_grad();
}
void neuron::n_output(value_array& input)
{
	weights_input = std::move(input* weights);  
	sum = std::move(weights_input.sum());
	sumwbias = std::move(sum + bias); 
	switch(neuron_activation){ 
		case TANH: 	
	 	out = std::move(sumwbias.tanh()); 
		break;
		case ELU:
	 	out = std::move(sumwbias.elu());
		case NONE:
	       break;	
	}
	return;
}
value& neuron::return_output() 
{
	if (neuron_activation== NONE)
		return sumwbias;
	if(neuron_activation!= NONE)
		return out;
	return out; 
}
layer::layer(){}
layer::layer(size_t amount, size_t input, activations act): m_size(amount), m_input(input), output(amount), m_func(act)
{
	m_neurons.reserve(amount);
       	for(size_t i=0; i<amount; i++){
		neuron temp(input,m_func);
		m_neurons.push_back(std::move(temp)); 
	} 
} 
void layer::l_output(value_array& input)
{
	std::vector<value> temp; 
	temp.reserve(m_size);
	std::for_each(m_neurons.begin(), m_neurons.end(), [&](neuron& x){
			x.n_output(input);
			temp.push_back(std::move(x.return_output().return_copy()));
			});
	output = temp; 
	return;
}
mlp::mlp(){}
mlp::mlp(int input_size ,std::initializer_list<int> neurons, std::initializer_list<activations> activation_func):layer_neurons(std::move(neurons)), functions(std::move(activation_func))
{
	layers.reserve(neurons.size()+ 1); 
	layers.push_back(std::move(layer(input_size, input_size, ELU)));	
	for(size_t i=1; i< neurons.size()+1; i++){
		layers.push_back(std::move(layer(layer_neurons[i-1], layer_neurons[i], functions[i])));
		}
}
